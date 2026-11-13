"""Hypothesis: the paper applied the FLUX output head to intermediate layers with a conditioning
vector (temb) different from the one the model used, while the final prediction is the true output.

FluxTransformer2DModel multiplies the timestep by 1000 before ``time_text_embed``; a probe that
rebuilds temb itself can easily pass the timestep on the wrong scale.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

import common  # noqa: F401  (puts the package on sys.path)
from depth_as_time import TransportTask
from depth_as_time.models.diffusers_adapter import DiffusersTransformerAdapter
from depth_as_time.readouts import fit_ridge_readout
from depth_as_time.text_to_image import load_pipeline, prepare_text_to_image
from explore_flux_protocol import PA, PS, fit
from run_flux import PROMPTS, log


@torch.no_grad()
def run(pipe, size: int, mode: str, seq: int, results: dict) -> None:
    n, batch, dev = (64, 8, "cuda:0") if size <= 512 else (32, 2, "cuda:0")
    prompts = [PROMPTS[0]] if mode == "single" else PROMPTS
    setup = prepare_text_to_image(pipe, prompts, num_steps=4, height=size, width=size, seed=0,
                                  device=dev, dtype=torch.bfloat16, max_sequence_length=seq)
    tr = pipe.transformer.to(dev).eval()
    adapter = DiffusersTransformerAdapter(tr, block_attrs=("single_transformer_blocks",))

    def temb_for(timestep_value: float, pooled: torch.Tensor) -> torch.Tensor:
        t = torch.full((pooled.shape[0],), timestep_value, device=dev, dtype=torch.bfloat16)
        return tr.time_text_embed(t, pooled)

    variants = {
        "true temb (captured)": lambda h, temb, pooled: tr.norm_out(h, temb),
        "timestep 1.0 (not x1000)": lambda h, temb, pooled: tr.norm_out(h, temb_for(1.0, pooled)),
        "timestep 1e6 (x1000 twice)": lambda h, temb, pooled: tr.norm_out(h, temb_for(1e6, pooled)),
        "timestep 0": lambda h, temb, pooled: tr.norm_out(h, temb_for(0.0, pooled)),
        "pooled text = zeros": lambda h, temb, pooled: tr.norm_out(
            h, temb_for(1000.0, torch.zeros_like(pooled))),
    }
    # Alternative explanation: the paper decoded layers with its trained linear probe (App. B.1)
    # instead of W_out. Fit the ridge probe on separate noise, apply it to every layer.
    def set_cond(idx):
        adapter.set_conditioning(**(setup.conditioning if mode == "single"
                                    else setup.conditioning_for(idx)))

    fit_noise = torch.randn(8, *setup.noise.shape[1:],
                            generator=torch.Generator().manual_seed(1)).to(dev, torch.bfloat16)
    ridge = fit_ridge_readout(adapter, fit_noise, TransportTask.one_step(8, device=dev),
                              batch_size=batch if size <= 512 else 1, on_batch=set_cond)
    weight = ridge.weight.to(dev)
    variants["trained ridge probe"] = lambda h, temb, pooled: None  # handled below
    xs = {k: [] for k in variants}
    fs, es = [], []
    gen = torch.Generator().manual_seed(0)
    for start in range(0, n, batch):
        idx = torch.arange(start, start + batch)
        cond = setup.conditioning if mode == "single" else setup.conditioning_for(idx)
        adapter.set_conditioning(**cond)
        cond_b = adapter._conditioning_for_batch(batch)
        z = torch.randn(batch, *setup.noise.shape[1:], generator=gen).to(dev, torch.bfloat16)
        task = TransportTask.one_step(batch, device=dev)
        hidden = adapter.hidden_states(z, task)
        temb, pooled = adapter._capture.temb, cond_b["pooled_projections"]
        for name, head in variants.items():
            if name == "trained ridge probe":
                v = torch.stack([F.layer_norm(h.float(), h.shape[-1:]) @ weight for h in hidden],
                                dim=1)
            else:
                v = torch.stack([tr.proj_out(head(h, temb, pooled)).float() for h in hidden],
                                dim=1)
            xs[name].append((z.float().unsqueeze(1) - v).cpu())
        fs.append(xs["true temb (captured)"][-1][:, -1])   # true model output
        es.append(z.float().cpu())
    f, e = torch.cat(fs).flatten(1), torch.cat(es).flatten(1)
    adapter.close()
    key = f"{size}_{mode}_seq{seq}"
    results[key] = {}
    log(f"{size}px {mode} seq{seq}: "
        f"paper sb00 [{PA[0]:.3f}, {PS[0]:.3f}]  sb20 [{PA[20]:.3f}, {PS[20]:.3f}]"
        f"  sb36 [{PA[-2]:.3f}, {PS[-2]:.3f}]  sb37 [{PA[-1]:.3f}, {PS[-1]:.3f}]")
    for name in variants:
        a, s = fit(torch.cat(xs[name]).flatten(2), f, e)
        results[key][name] = {"alpha": a.tolist(), "sigma": s.tolist()}
        log(f"{name:27s} alpha err {(np.abs(a - PA) / np.abs(PA)).mean() * 100:6.1f}%  sigma err "
            f"{(np.abs(s - PS) / np.abs(PS)).mean() * 100:6.1f}%  sb00 [{a[0]:.3f}, {s[0]:.3f}]  "
            f"sb20 [{a[20]:.3f}, {s[20]:.3f}]  sb36 [{a[-2]:.3f}, {s[-2]:.3f}]  "
            f"sb37 [{a[-1]:.3f}, {s[-1]:.3f}]")


def main() -> None:
    """Arguments: one or more ``pixels:prompt_mode:t5_tokens`` triples, e.g. 512:single:256."""
    pipe = load_pipeline("black-forest-labs/FLUX.1-schnell", torch.bfloat16)
    path = common.DATA / "flux_head_conditioning.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for spec in sys.argv[1:]:
        size, mode, seq = spec.split(":")
        run(pipe, int(size), mode, int(seq), results)
        common.save_json(results, path)


if __name__ == "__main__":
    main()
