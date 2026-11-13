"""Hypothesis: the model input is s * eps, while eps is used to decode layers and as regressor."""

from __future__ import annotations

import sys

import numpy as np
import torch

import common  # noqa: F401  (puts the package on sys.path)
from depth_as_time import TransportTask
from depth_as_time.models.diffusers_adapter import DiffusersTransformerAdapter
from depth_as_time.text_to_image import load_pipeline, prepare_text_to_image
from explore_flux_protocol import PA, PS, fit
from run_flux import PROMPTS, log


@torch.no_grad()
def main() -> None:
    size, n, batch, dev = int(sys.argv[1]), 64, 8, "cuda:0"
    pipe = load_pipeline("black-forest-labs/FLUX.1-schnell", torch.bfloat16)
    setup = prepare_text_to_image(pipe, [PROMPTS[0]], num_steps=4, height=size, width=size,
                                  seed=0, device=dev, dtype=torch.bfloat16)
    tr = pipe.transformer.to(dev).eval()
    adapter = DiffusersTransformerAdapter(tr, block_attrs=("single_transformer_blocks",))
    adapter.set_conditioning(**setup.conditioning)
    for s in (1.0, 1.1, 1.16, 1.2, 1.3):
        gen = torch.Generator().manual_seed(0)
        xs, fs, es = [], [], []
        for _ in range(0, n, batch):
            eps = torch.randn(batch, *setup.noise.shape[1:], generator=gen).to(dev)
            z = (s * eps).to(torch.bfloat16)
            task = TransportTask.one_step(batch, device=dev)
            hidden = adapter.hidden_states(z, task)
            v = torch.stack([adapter.readout(h, task).float() for h in hidden], dim=1)
            xs.append((eps.unsqueeze(1) - v).cpu())      # layers decoded with the recorded eps
            fs.append((z.float() - v[:, -1]).cpu())      # true model output
            es.append(eps.cpu())
        x, f, e = torch.cat(xs).flatten(2), torch.cat(fs).flatten(1), torch.cat(es).flatten(1)
        a, sg = fit(x, f, e)
        log(f"s={s:4.2f}: alpha err {(np.abs(a - PA) / np.abs(PA)).mean() * 100:5.1f}%  sigma err "
            f"{(np.abs(sg - PS) / np.abs(PS)).mean() * 100:5.1f}%  sb00 [{a[0]:.3f}, {sg[0]:.3f}]  "
            f"sb20 [{a[20]:.3f}, {sg[20]:.3f}]  sb36 [{a[-2]:.3f}, {sg[-2]:.3f}]  "
            f"sb37 [{a[-1]:.3f}, {sg[-1]:.3f}]")
    log(f"paper : sb00 [{PA[0]:.3f}, {PS[0]:.3f}]  sb20 [{PA[20]:.3f}, {PS[20]:.3f}]  "
        f"sb36 [{PA[-2]:.3f}, {PS[-2]:.3f}]  sb37 [{PA[-1]:.3f}, {PS[-1]:.3f}]")


if __name__ == "__main__":
    main()
