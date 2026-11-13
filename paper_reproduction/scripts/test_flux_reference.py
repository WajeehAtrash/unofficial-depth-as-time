"""Hypothesis test: did the paper fit FLUX's mixing schedule against the 4-step sample?

The paper's last FLUX block has (alpha, sigma) = (0.96, -0.16) instead of the (1, 0) that a fit
against the one-step output must give. Here the same one-step layer decodes (the first NFE of the
4-step run; FLUX-schnell ignores r, so it is the same network call) are fitted against
(a) the one-step output x_0^L and (b) the final 4-step sample.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from common import DATA, ROOT, save_json
from depth_as_time import LayerwiseProbe, probe_sampling_trajectory
from depth_as_time.mixing import MixingStatistics, evaluate_mixing
from depth_as_time.models.diffusers_adapter import DiffusersTransformerAdapter
from depth_as_time.text_to_image import load_pipeline, prepare_text_to_image
from run_flux import PROMPTS, log

N, BATCH, TRAIN = 64, 2, 45


@torch.no_grad()
def main() -> None:
    dev = "cuda"
    pipe = load_pipeline("black-forest-labs/FLUX.1-schnell", torch.bfloat16)
    setup = prepare_text_to_image(pipe, PROMPTS, num_steps=4, height=1024, width=1024, seed=0,
                                  device=dev, dtype=torch.bfloat16)
    pipe.transformer.to(dev).eval()
    adapter = DiffusersTransformerAdapter(pipe.transformer,
                                          block_attrs=("single_transformer_blocks",))
    probe = LayerwiseProbe(adapter)
    stats = {k: (MixingStatistics(38), MixingStatistics(38)) for k in ("one_step", "four_step")}
    g = torch.Generator().manual_seed(0)
    for start in range(0, N, BATCH):
        idx = torch.arange(start, start + BATCH)
        adapter.set_conditioning(**setup.conditioning_for(idx))
        eps = torch.randn(BATCH, *setup.noise.shape[1:], generator=g).to(dev, torch.bfloat16)
        steps = probe_sampling_trajectory(probe, eps, setup.sigmas)
        clean = steps[0].clean  # one-step layer decodes, t = 1
        refs = {"one_step": clean[-1], "four_step": steps[-1].next_state.float()}
        train = start < TRAIN
        for key, ref in refs.items():
            stats[key][0 if train else 1].update(clean, ref, eps)
        log(f"{start + BATCH}/{N}")
    paper = json.loads((ROOT / "paper_data/paper_vector_data.json").read_text())["coeffs_flux_sb"]
    out = {}
    for key, (tr, te) in stats.items():
        m = evaluate_mixing(tr, te)
        a, s = m.alpha.numpy(), m.sigma.numpy()
        pa, ps = np.array(paper["alpha"]), np.array(paper["sigma"])
        out[key] = {
            "alpha": m.alpha, "sigma": m.sigma, "mean_R": m.mean_held_out_r,
            "alpha_mean_rel_pct": float((np.abs(a - pa) / np.abs(pa)).mean() * 100),
            "sigma_mean_rel_pct": float((np.abs(s - ps)[1:-1] / np.abs(ps)[1:-1]).mean() * 100),
            "last_block": [float(a[-1]), float(s[-1])],
        }
        log(f"{key}: alpha err {out[key]['alpha_mean_rel_pct']:.1f}%, sigma err "
            f"{out[key]['sigma_mean_rel_pct']:.1f}%, last block {out[key]['last_block']}, "
            f"sb00 ({a[0]:.2f}, {s[0]:.2f}) vs paper ({pa[0]:.2f}, {ps[0]:.2f})")
    save_json(out, DATA / "flux_reference_test.json")


if __name__ == "__main__":
    main()
