"""Bug hunt for the FLUX mixing-coefficient gap (Fig. B.2): trace every stage with a direct test.

A. Adapter vs raw transformer / stock pipeline: see ``localize_pipeline_gap.py``.
B. Output head applied in float32 instead of bf16.
C. Fit conventions: intercept, centering.
D. Whole transformer in float32 (split over two GPUs) vs bf16 on the same trajectories.

Results are written to ``data/flux_diagnosis.json``.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from common import DATA, ROOT, free_cuda, save_json
from depth_as_time import TransportTask
from depth_as_time.models.diffusers_adapter import DiffusersTransformerAdapter
from depth_as_time.text_to_image import load_pipeline, prepare_text_to_image
from run_flux import PROMPTS, log

N = 16
MODEL = "black-forest-labs/FLUX.1-schnell"
PAPER = json.loads((ROOT / "paper_data/paper_vector_data.json").read_text())["coeffs_flux_sb"]


def fit(x: torch.Tensor, f: torch.Tensor, e: torch.Tensor, intercept: bool = False):
    """Least squares x_i ~ alpha f + sigma e (+ c) over all trajectories; x is (N, L, D)."""
    cols = [f.flatten().double(), e.flatten().double()]
    if intercept:
        cols.append(torch.ones_like(cols[0]))
    a = torch.stack(cols, dim=1)
    gram_inv = torch.linalg.pinv(a.T @ a)
    coef = torch.stack([gram_inv @ (a.T @ x[:, i].flatten().double()) for i in range(x.shape[1])])
    return coef[:, 0].numpy(), coef[:, 1].numpy()


def errors(alpha, sigma) -> dict:
    pa, ps = np.array(PAPER["alpha"]), np.array(PAPER["sigma"])
    return {
        "alpha_mean_rel_pct": float((np.abs(alpha - pa) / np.abs(pa)).mean() * 100),
        "sigma_mean_rel_pct": float((np.abs(sigma - ps) / np.abs(ps)).mean() * 100),
        "sb00": [float(alpha[0]), float(sigma[0])],
        "sb37": [float(alpha[-1]), float(sigma[-1])],
    }


def collect(adapter, setup, noise, head=None):
    """Layerwise clean decodes (N, L, D) on the CPU in float32. ``head`` overrides the readout."""
    out = []
    for i in range(noise.shape[0]):
        adapter.set_conditioning(**setup.conditioning_for(torch.tensor([i])))
        z = noise[i : i + 1]
        task = TransportTask.one_step(1, device=z.device)
        hidden = adapter.hidden_states(z, task)
        ro = adapter.readout_tokens if head is None else head
        v = torch.stack([ro(h, task).float().cpu() for h in hidden])[:, 0]
        out.append(z[0].float().cpu() - v)  # t = 1
        del hidden
        free_cuda()
    return torch.stack(out).flatten(2)  # (N, L, D)


@torch.no_grad()
def main() -> None:
    dev = "cuda:0"
    res: dict = {}
    pipe = load_pipeline(MODEL, torch.bfloat16)
    setup = prepare_text_to_image(pipe, PROMPTS[:N], num_steps=4, height=1024, width=1024, seed=0,
                                  device=dev, dtype=torch.bfloat16)
    noise = torch.randn(N, *setup.noise.shape[1:], generator=torch.Generator().manual_seed(0))
    noise = noise.to(dev, torch.bfloat16)
    eps = noise.float().cpu().flatten(1)
    tr = pipe.transformer.to(dev).eval()
    adapter = DiffusersTransformerAdapter(tr, block_attrs=("single_transformer_blocks",))

    # ---- baseline (bf16), same code path as run_flux.py
    x = collect(adapter, setup, noise)
    a0, s0 = fit(x, x[:, -1], eps)
    res["baseline_bf16"] = errors(a0, s0)
    log(f"baseline {res['baseline_bf16']}")

    # (A, the stock-pipeline check, lives in localize_pipeline_gap.py.)
    pipe.text_encoder, pipe.text_encoder_2 = None, None
    free_cuda()

    # ---- B: output head in float32
    # Cast the head in place for this pass (weights are bf16, so the round trip is lossless).
    # deepcopy is not an option: the adapter's hook on norm_out would drag the whole model along.
    def head32(h, task):
        tr.norm_out.float(), tr.proj_out.float()
        out = tr.proj_out(tr.norm_out(h.float(), adapter._capture.temb.float()))
        tr.norm_out.to(torch.bfloat16), tr.proj_out.to(torch.bfloat16)
        return out

    xb = collect(adapter, setup, noise, head=head32)
    ab, sb = fit(xb, xb[:, -1], eps)
    res["B_head_float32"] = {**errors(ab, sb),
                             "max_coeff_change_vs_baseline": float(max(np.abs(ab - a0).max(),
                                                                       np.abs(sb - s0).max()))}
    log(f"B head fp32 {res['B_head_float32']}")

    # ---- C: fit conventions on the baseline decodes
    f = x[:, -1]
    ai, si = fit(x, f, eps, intercept=True)
    res["C_intercept"] = errors(ai, si)
    shape = (N, -1, 4096, 64)
    xc = x.reshape(shape)
    ec = eps.reshape(N, 4096, 64)
    xch = (xc - xc.mean(dim=2, keepdim=True)).flatten(2)       # remove per-channel mean
    ech = (ec - ec.mean(dim=1, keepdim=True)).flatten(1)
    ac, sc = fit(xch, xch[:, -1], ech)
    res["C_channel_centered"] = errors(ac, sc)
    xtk = (xc - xc.mean(dim=3, keepdim=True)).flatten(2)       # remove per-token mean
    etk = (ec - ec.mean(dim=2, keepdim=True)).flatten(1)
    at, st = fit(xtk, xtk[:, -1], etk)
    res["C_token_centered"] = errors(at, st)
    log(f"C intercept {res['C_intercept']}")
    log(f"C channel-centered {res['C_channel_centered']}")
    log(f"C token-centered {res['C_token_centered']}")
    save_json(res, DATA / "flux_diagnosis.json")

    # ---- D: full float32 transformer over two GPUs
    adapter.close()
    cond32 = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
              for k, v in setup.conditioning.items()}
    setup.conditioning = cond32
    # Drop the bf16 model (names are rebound, not deleted: head32 above still refers to them).
    pipe = tr = adapter = None
    free_cuda()
    from diffusers import FluxTransformer2DModel

    from huggingface_hub import snapshot_download

    tr32 = FluxTransformer2DModel.from_pretrained(
        snapshot_download(MODEL, local_files_only=True), subfolder="transformer",
        torch_dtype=torch.float32, device_map="auto",
        max_memory={0: "40GiB", 1: "40GiB"},
    ).eval()
    adapter32 = DiffusersTransformerAdapter(tr32, block_attrs=("single_transformer_blocks",))

    def head_dev(h, task):  # the head lives on the last device
        w = tr32.proj_out.weight
        return tr32.proj_out(tr32.norm_out(h.to(w.device), adapter32._capture.temb.to(w.device)))

    xd = collect(adapter32, setup, noise.float(), head=head_dev)
    ad, sd = fit(xd, xd[:, -1], eps)
    res["D_full_float32"] = {
        **errors(ad, sd),
        "max_coeff_change_vs_bf16": float(max(np.abs(ad - a0).max(), np.abs(sd - s0).max())),
        "final_output_rel_l2_vs_bf16": float((xd[:, -1] - x[:, -1]).norm() / x[:, -1].norm()),
    }
    log(f"D full fp32 {res['D_full_float32']}")
    save_json(res, DATA / "flux_diagnosis.json")


if __name__ == "__main__":
    main()
