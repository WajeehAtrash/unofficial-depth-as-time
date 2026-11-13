"""Prepare text-to-image diffusers pipelines (FLUX, SD3 and SD3.5) for layerwise probing.

The functions here do the pipeline-side work the probe needs: encode prompts, draw initial
noise in the transformer's input layout, read the sampler's sigma schedule, and decode latents
to images with the VAE for visualization. The transformer itself is wrapped by
:class:`~depth_as_time.models.diffusers_adapter.DiffusersTransformerAdapter`.
"""

from __future__ import annotations

import copy
import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from depth_as_time.models.diffusers_adapter import infer_family

# Hugging Face repository ids of the models used in the paper and of common turbo models.
MODEL_PRESETS: dict[str, str] = {
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    "sd35-large-turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
}

_TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3")
_BATCHED_KEYS = ("encoder_hidden_states", "pooled_projections", "guidance")


@dataclass
class TextToImageSetup:
    """Everything needed to probe a text-to-image model on a set of prompts.

    Attributes:
        family: ``"flux"`` or ``"sd3"``.
        conditioning: Keyword arguments for the transformer with one batch entry per prompt.
        noise: Initial noise ``z_1``, one draw per prompt, in the transformer input layout.
        sigmas: Sampler schedule ``[1, ..., 0]`` with ``num_steps + 1`` entries.
        decode: Maps transformer-layout latents ``(B, ...)`` to RGB images in ``[0, 1]``.
    """

    family: str
    conditioning: dict[str, Any]
    noise: Tensor
    sigmas: list[float]
    decode: Callable[[Tensor], Tensor]

    @property
    def num_prompts(self) -> int:
        """Number of prompts, which is also the number of noise draws in :attr:`noise`."""
        return self.noise.shape[0]

    def conditioning_for(self, indices: Tensor) -> dict[str, Any]:
        """Conditioning for the given trajectory indices, cycling through the prompts."""
        rows = indices.remainder(self.num_prompts)
        selected = {}
        for key, value in self.conditioning.items():
            is_batched = key in _BATCHED_KEYS and torch.is_tensor(value) and value.shape[0] > 1
            selected[key] = value[rows.to(value.device)] if is_batched else value
        return selected


def load_pipeline(model_id: str, dtype: torch.dtype = torch.bfloat16) -> Any:
    """Load a diffusers pipeline on CPU. Gated models need ``huggingface-cli login`` first."""
    from diffusers import DiffusionPipeline

    return DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)


def sampler_sigmas(
    scheduler: Any, num_steps: int, image_seq_len: int, sigmas: Sequence[float] | None = None
) -> list[float]:
    """Sigma schedule a flow-matching Euler scheduler would use, including the final 0.

    The scheduler is copied, so the pipeline is left untouched. If it uses dynamic shifting
    (FLUX.1 dev), the resolution dependent shift ``mu`` is computed as in the pipeline.
    """
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    scheduler = copy.deepcopy(scheduler)
    config = scheduler.config
    kwargs: dict[str, Any] = {}
    if config.get("use_dynamic_shifting", False):
        kwargs["mu"] = calculate_shift(
            image_seq_len,
            config.get("base_image_seq_len", 256),
            config.get("max_image_seq_len", 4096),
            config.get("base_shift", 0.5),
            config.get("max_shift", 1.15),
        )
    scheduler.set_timesteps(num_steps, device="cpu", sigmas=sigmas, **kwargs)
    return [float(s) for s in scheduler.sigmas]


def unpack_flux_latents(latents: Tensor, height: int, width: int, vae_scale_factor: int) -> Tensor:
    """Unpack FLUX tokens ``(B, N, 4C)`` into a latent image ``(B, C, H/8, W/8)``."""
    batch, _, channels = latents.shape
    height = 2 * (height // (vae_scale_factor * 2))
    width = 2 * (width // (vae_scale_factor * 2))
    latents = latents.view(batch, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch, channels // 4, height, width)


def _make_vae_decoder(
    vae: Any, to_latent_image: Callable[[Tensor], Tensor], chunk_size: int = 2
) -> Callable[[Tensor], Tensor]:
    scaling = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", None) or 0.0

    @torch.no_grad()
    def decode(latents: Tensor) -> Tensor:
        images = []
        for chunk in latents.split(chunk_size):
            latent_image = to_latent_image(chunk) / scaling + shift
            image = vae.decode(latent_image.to(device=vae.device, dtype=vae.dtype)).sample
            images.append(((image.float() + 1.0) / 2.0).clamp(0.0, 1.0).cpu())
        return torch.cat(images)

    return decode


def _encode_with_offload(pipe: Any, device: torch.device | str, encode: Callable[[], Any]) -> Any:
    """Run ``encode`` with the text encoders on ``device``, then send them back to the CPU."""
    encoders = [getattr(pipe, a) for a in _TEXT_ENCODER_ATTRS if getattr(pipe, a, None) is not None]
    for encoder in encoders:
        encoder.to(device)
    try:
        with torch.no_grad():
            return encode()
    finally:
        for encoder in encoders:
            encoder.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def prepare_text_to_image(
    pipe: Any,
    prompts: Sequence[str],
    num_steps: int,
    height: int = 1024,
    width: int = 1024,
    seed: int = 0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_sequence_length: int = 256,
    guidance_scale: float = 3.5,
) -> TextToImageSetup:
    """Encode prompts and draw noise for a FLUX or SD3 pipeline.

    Text encoders are moved to ``device`` only while encoding, so a large T5 and the
    transformer do not need to fit on the GPU together. The VAE is moved to ``device`` and
    stays there for decoding. Moving the transformer is left to the caller.

    Args:
        pipe: A loaded ``FluxPipeline`` or ``StableDiffusion3Pipeline``.
        prompts: One prompt per noise draw.
        num_steps: Sampler steps used to compute the sigma schedule.
        height: Image height in pixels.
        width: Image width in pixels.
        seed: Seed for the initial noise.
        device: Device for encoding, noise and the VAE.
        dtype: Dtype of embeddings and noise.
        max_sequence_length: T5 sequence length (FLUX.1 schnell is trained with 256).
        guidance_scale: Guidance embedding value, used only by guidance-distilled FLUX.1 dev.
    """
    family = infer_family(pipe.transformer)
    generator = torch.Generator(device=device).manual_seed(seed)
    batch = len(prompts)
    pipe.vae.to(device)
    if family == "flux":
        return _prepare_flux(
            pipe,
            prompts,
            batch,
            num_steps,
            height,
            width,
            generator,
            device,
            dtype,
            max_sequence_length,
            guidance_scale,
        )
    return _prepare_sd3(
        pipe,
        prompts,
        batch,
        num_steps,
        height,
        width,
        generator,
        device,
        dtype,
        max_sequence_length,
    )


def _prepare_flux(
    pipe, prompts, batch, num_steps, height, width, generator, device, dtype, max_len, guidance
) -> TextToImageSetup:
    prompt_embeds, pooled, text_ids = _encode_with_offload(
        pipe,
        device,
        lambda: pipe.encode_prompt(
            prompt=list(prompts), prompt_2=None, device=device, max_sequence_length=max_len
        ),
    )
    channels = pipe.transformer.config.in_channels // 4
    noise, img_ids = pipe.prepare_latents(batch, channels, height, width, dtype, device, generator)
    conditioning: dict[str, Any] = {
        "encoder_hidden_states": prompt_embeds.to(dtype),
        "pooled_projections": pooled.to(dtype),
        "txt_ids": text_ids,
        "img_ids": img_ids,
    }
    if pipe.transformer.config.guidance_embeds:
        conditioning["guidance"] = torch.full((batch,), guidance, device=device, dtype=dtype)

    sigmas = sampler_sigmas(
        pipe.scheduler, num_steps, noise.shape[1], np.linspace(1.0, 1.0 / num_steps, num_steps)
    )
    scale = pipe.vae_scale_factor
    decode = _make_vae_decoder(pipe.vae, lambda x: unpack_flux_latents(x, height, width, scale))
    return TextToImageSetup("flux", conditioning, noise, sigmas, decode)


def _prepare_sd3(
    pipe, prompts, batch, num_steps, height, width, generator, device, dtype, max_len
) -> TextToImageSetup:
    prompt_embeds, _, pooled, _ = _encode_with_offload(
        pipe,
        device,
        lambda: pipe.encode_prompt(
            prompt=list(prompts),
            prompt_2=None,
            prompt_3=None,
            device=device,
            do_classifier_free_guidance=False,
            max_sequence_length=max_len,
        ),
    )
    scale = pipe.vae_scale_factor
    shape = (batch, pipe.transformer.config.in_channels, height // scale, width // scale)
    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    conditioning = {
        "encoder_hidden_states": prompt_embeds.to(dtype),
        "pooled_projections": pooled.to(dtype),
    }
    patch = pipe.transformer.config.patch_size
    image_seq_len = (shape[2] // patch) * (shape[3] // patch)
    sigmas = sampler_sigmas(pipe.scheduler, num_steps, image_seq_len)
    decode = _make_vae_decoder(pipe.vae, lambda x: x)
    return TextToImageSetup("sd3", conditioning, noise, sigmas, decode)
