"""Localize why our one-step FLUX output differs by ~12% from a stock FluxPipeline call.

Finding: the adapter equals a raw transformer call exactly; the difference comes from bf16 T5
embeddings, which change by ~3% when a prompt is encoded alone instead of in a batch.
"""

from __future__ import annotations

import torch

from depth_as_time import TransportTask
from depth_as_time.models.diffusers_adapter import DiffusersTransformerAdapter
from depth_as_time.text_to_image import load_pipeline, prepare_text_to_image
from run_flux import PROMPTS, log


def rel(a, b):
    a, b = a.float().cpu().flatten(), b.float().cpu().flatten()
    return float((a - b).norm() / b.norm())


@torch.no_grad()
def main() -> None:
    dev = "cuda:0"
    pipe = load_pipeline("black-forest-labs/FLUX.1-schnell", torch.bfloat16)
    kw = dict(num_steps=4, height=1024, width=1024, seed=0, device=dev, dtype=torch.bfloat16)
    setup16 = prepare_text_to_image(pipe, PROMPTS[:16], **kw)   # as in the experiments
    setup1 = prepare_text_to_image(pipe, PROMPTS[:1], **kw)     # prompt encoded alone
    c16 = setup16.conditioning_for(torch.tensor([0]))
    c1 = setup1.conditioning
    for k in ("encoder_hidden_states", "pooled_projections", "txt_ids", "img_ids"):
        log(f"conditioning {k}: batch-of-16 vs alone rel diff {rel(c16[k], c1[k]):.2e}, "
            f"shape {tuple(c16[k].shape)} vs {tuple(c1[k].shape)}")
    z = torch.randn(1, *setup16.noise.shape[1:], generator=torch.Generator().manual_seed(0))
    z = z.to(dev, torch.bfloat16)
    tr = pipe.transformer.to(dev).eval()
    task = TransportTask.one_step(1, device=dev)

    def ours(cond, block_attrs=None):
        ad = DiffusersTransformerAdapter(tr, block_attrs=block_attrs)
        ad.set_conditioning(**cond)
        h = ad.hidden_states(z, task)
        v = ad.readout(h[-1], task).float()
        ad.close()
        return z.float() - v

    def raw(cond, timestep):
        v = tr(hidden_states=z, timestep=timestep, return_dict=False, **cond)[0].float()
        return z.float() - v

    x16 = ours(c16)
    log(f"adapter vs raw transformer call, same inputs: {rel(x16, raw(c16, task.t)):.2e}")
    log(f"repeat of the same adapter call: {rel(ours(c16), x16):.2e}")
    log(f"prompt encoded alone vs in a batch of 16: {rel(ours(c1), x16):.2e}")
    log(f"timestep passed as bf16 vs float32: "
        f"{rel(raw(c16, task.t.to(torch.bfloat16)), raw(c16, task.t)):.2e}")


if __name__ == "__main__":
    main()
