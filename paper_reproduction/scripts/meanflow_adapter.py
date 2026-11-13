"""LayerwiseModel adapter for the SiT MeanFlow models of MeanFlow-PyTorch.

The SiT forward is ``c = t_emb(t) + h_emb(t - r) + y_emb(y)``, ``x = blocks(x, c)``,
``out = unpatchify(final_layer(x, c))``. Time convention matches this package: ``t = 1`` is
noise, ``r`` is the endpoint, the second embedder receives the interval ``h = t - r``.
"""

from __future__ import annotations

import sys

import torch
from torch import Tensor

from common import MEANFLOW_REPO
from depth_as_time.models.base import LayerwiseModel
from depth_as_time.task import TransportTask

sys.path.insert(0, str(MEANFLOW_REPO))
from models.sit_meanflow import SiT_models, modulate  # noqa: E402

LATENT_SCALE = 0.18215


def load_meanflow(ckpt: str, model_name: str = "SiT-B/4", device: str = "cuda") -> torch.nn.Module:
    model = SiT_models[model_name](
        input_size=32,
        num_classes=1000,
        use_cfg=True,
        class_dropout_prob=0.1,
        encoder_depth=0,
        fused_attn=False,
        qk_norm=False,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "ema" in state:
        state = state["ema"]
    model.load_state_dict(state)
    return model.to(device).eval()


class MeanFlowAdapter(LayerwiseModel):
    """Expose the 12 SiT blocks of a MeanFlow model to the layerwise probe."""

    def __init__(self, sit: torch.nn.Module) -> None:
        super().__init__()
        self.sit = sit

    @property
    def num_layers(self) -> int:
        return len(self.sit.blocks)

    def conditioning(self, task: TransportTask, label: Tensor) -> Tensor:
        sit = self.sit
        return sit.t_embedder(task.t) + sit.h_embedder(task.interval) + sit.y_embedder(label, False)

    @torch.no_grad()
    def hidden_states(self, z: Tensor, task: TransportTask, label: Tensor | None = None) -> list:
        sit = self.sit
        c = self.conditioning(task, label)
        x = sit.x_embedder(z) + sit.pos_embed
        states = []
        for block in sit.blocks:
            x = block(x, c)
            states.append(x)
        return states

    def readout_tokens(self, h: Tensor, task: TransportTask, label: Tensor | None = None) -> Tensor:
        return self.sit.final_layer(h, self.conditioning(task, label))

    def tokens_to_output(self, tokens: Tensor) -> Tensor:
        return self.sit.unpatchify(tokens)


class RMSOnlyReadout:
    """The model head with LayerNorm replaced by RMS normalization (no centering), App. B.2."""

    def __init__(self, adapter: MeanFlowAdapter) -> None:
        self.adapter = adapter

    def __call__(self, h: Tensor, task: TransportTask, label: Tensor | None = None) -> Tensor:
        final = self.adapter.sit.final_layer
        c = self.adapter.conditioning(task, label)
        shift, scale = final.adaLN_modulation(c).chunk(2, dim=-1)
        x = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        x = final.linear(modulate(x, shift, scale))
        return self.adapter.tokens_to_output(x)


def make_vae_decoder(device: str = "cuda"):
    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()

    @torch.no_grad()
    def decode(latents: Tensor, chunk: int = 16) -> Tensor:
        out = []
        for part in latents.split(chunk):
            img = vae.decode(part.to(device=device, dtype=torch.float32) / LATENT_SCALE).sample
            out.append(((img.float() + 1) / 2).clamp(0, 1).cpu())
        return torch.cat(out)

    return decode
