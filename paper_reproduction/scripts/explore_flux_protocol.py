"""Search the unstated parts of the paper's FLUX mixing protocol (Fig. B.2).

The paper says only: "256 trajectories from a single network evaluation with a fixed seed". Its
Fig. 4 residual maps are 32 cells wide, which means 256 px (latent grid) or 512 px (token grid)
images. This script varies resolution, prompt set and text length, and for each configuration
fits the coefficients against several definitions of the "final prediction".

Per-trajectory decodes are saved (float16) so further hypotheses can be tested offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import DATA, ROOT, free_cuda, save_json
from depth_as_time import TransportTask
from depth_as_time.models.diffusers_adapter import DiffusersTransformerAdapter
from depth_as_time.text_to_image import load_pipeline, prepare_text_to_image, unpack_flux_latents
from run_flux import PROMPTS, log

PAPER = json.loads((ROOT / "paper_data/paper_vector_data.json").read_text())["coeffs_flux_sb"]
PA, PS = np.array(PAPER["alpha"]), np.array(PAPER["sigma"])

# name: (pixels, prompt mode, T5 length, trajectories, batch)
CONFIGS = {
    "256_single_seq256": (256, "single", 256, 128, 16),
    "256_multi_seq256": (256, "multi", 256, 128, 16),
    "256_single_seq512": (256, "single", 512, 128, 16),
    "256_empty_seq256": (256, "empty", 256, 128, 16),
    "512_single_seq256": (512, "single", 256, 64, 8),
    "512_single_seq512": (512, "single", 512, 64, 8),
    "1024_single_seq256": (1024, "single", 256, 32, 2),
}


def fit(x: torch.Tensor, f: torch.Tensor, e: torch.Tensor):
    """Least squares x_i ~ alpha f + sigma e pooled over trajectories; x is (N, L, D)."""
    a = torch.stack([f.flatten().double(), e.flatten().double()], dim=1)
    pinv = torch.linalg.pinv(a.T @ a)
    coef = torch.stack([pinv @ (a.T @ x[:, i].flatten().double()) for i in range(x.shape[1])])
    return coef[:, 0].numpy(), coef[:, 1].numpy()


def score(alpha, sigma) -> dict:
    return {
        "alpha_err_pct": float((np.abs(alpha - PA) / np.abs(PA)).mean() * 100),
        "sigma_err_pct": float((np.abs(sigma - PS) / np.abs(PS)).mean() * 100),
        "sb00": [round(float(alpha[0]), 3), round(float(sigma[0]), 3)],
        "sb36": [round(float(alpha[-2]), 3), round(float(sigma[-2]), 3)],
        "sb37": [round(float(alpha[-1]), 3), round(float(sigma[-1]), 3)],
        "alpha": alpha.tolist(), "sigma": sigma.tolist(),
    }


@torch.no_grad()
def run_config(pipe, name: str, dev: str, out_dir: Path) -> dict:
    size, mode, seq, n, batch = CONFIGS[name]
    prompts = {"single": [PROMPTS[0]], "multi": PROMPTS, "empty": [""]}[mode]
    setup = prepare_text_to_image(pipe, prompts, num_steps=4, height=size, width=size, seed=0,
                                  device=dev, dtype=torch.bfloat16, max_sequence_length=seq)
    tr = pipe.transformer.to(dev).eval()
    adapter = DiffusersTransformerAdapter(tr, block_attrs=("single_transformer_blocks",))
    shape = tuple(setup.noise.shape[1:])
    gen = torch.Generator().manual_seed(0)
    xs, es, four, rt = [], [], [], []
    vae, scale = pipe.vae, pipe.vae_scale_factor
    sf, sh = vae.config.scaling_factor, vae.config.shift_factor
    for start in range(0, n, batch):
        idx = torch.arange(start, start + batch)
        cond = setup.conditioning_for(idx) if mode == "multi" else setup.conditioning
        adapter.set_conditioning(**cond)
        z = torch.randn(batch, *shape, generator=gen).to(dev, torch.bfloat16)
        task = TransportTask.one_step(batch, device=dev)
        hidden = adapter.hidden_states(z, task)
        v = torch.stack([adapter.readout(h, task).float() for h in hidden], dim=1)  # (B, L, N, C)
        x = z.float().unsqueeze(1) - v
        xs.append(x.cpu())
        es.append(z.float().cpu())
        del hidden
        # reference variant: final sample of the 4-step sampler from the same noise
        zt, sig = z, setup.sigmas
        cond_b = adapter._conditioning_for_batch(batch)
        for k in range(4):
            t = torch.full((batch,), sig[k], device=dev, dtype=torch.bfloat16)
            vk = tr(hidden_states=zt, timestep=t, return_dict=False, **cond_b)[0]
            zt = (zt.float() - (sig[k] - sig[k + 1]) * vk.float()).to(torch.bfloat16)
        four.append(zt.float().cpu())
        # reference variant: VAE decode -> encode round trip of the one-step output
        lat = unpack_flux_latents(x[:, -1], size, size, scale) / sf + sh
        img = vae.decode(lat.to(vae.dtype)).sample
        back = (vae.encode(img).latent_dist.mode().float() - sh) * sf
        b, c, hh, ww = back.shape
        back = back.view(b, c, hh // 2, 2, ww // 2, 2).permute(0, 2, 4, 1, 3, 5)
        rt.append(back.reshape(b, (hh // 2) * (ww // 2), c * 4).cpu())
        free_cuda()
    adapter.close()
    x = torch.cat(xs).flatten(2)          # (N, L, D)
    e = torch.cat(es).flatten(1)
    four, rt = torch.cat(four).flatten(1), torch.cat(rt).flatten(1)
    res = {
        "config": {"pixels": size, "prompts": mode, "t5_tokens": seq, "trajectories": n},
        "ref_one_step": score(*fit(x, x[:, -1], e)),
        "ref_four_step": score(*fit(x, four, e)),
        "ref_vae_roundtrip": score(*fit(x, rt, e)),
    }
    torch.save({"x": x.half(), "eps": e.half(), "four_step": four.half(), "roundtrip": rt.half()},
               out_dir / f"{name}.pt")
    for key in ("ref_one_step", "ref_four_step", "ref_vae_roundtrip"):
        r = res[key]
        log(f"{name:20s} {key:18s} alpha err {r['alpha_err_pct']:5.1f}%  sigma err "
            f"{r['sigma_err_pct']:5.1f}%  sb00 {r['sb00']}  sb36 {r['sb36']}  sb37 {r['sb37']}")
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="*", default=list(CONFIGS))
    p.add_argument("--tensor-dir", required=True, help="Where to save per-trajectory decodes.")
    args = p.parse_args()
    out_dir = Path(args.tensor_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"paper: sb00 [{PA[0]:.3f}, {PS[0]:.3f}]  sb36 [{PA[-2]:.3f}, {PS[-2]:.3f}]  "
        f"sb37 [{PA[-1]:.3f}, {PS[-1]:.3f}]")
    pipe = load_pipeline("black-forest-labs/FLUX.1-schnell", torch.bfloat16)
    path = DATA / "flux_protocol_search.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in args.configs:
        results[name] = run_config(pipe, name, "cuda:0", out_dir)
        save_json(results, path)


if __name__ == "__main__":
    main()
