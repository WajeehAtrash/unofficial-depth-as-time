import pytest
import torch

pytest.importorskip("diffusers")

from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline  # noqa: E402

from depth_as_time.text_to_image import (  # noqa: E402
    TextToImageSetup,
    sampler_sigmas,
    unpack_flux_latents,
)


def test_unpack_matches_diffusers_reference():
    latents = torch.randn(2, 16 * 16, 64)
    ours = unpack_flux_latents(latents, height=256, width=256, vae_scale_factor=8)
    reference = FluxPipeline._unpack_latents(latents, 256, 256, 8)
    assert ours.shape == (2, 16, 32, 32)
    assert torch.equal(ours, reference)


def test_sampler_sigmas_are_a_shifted_decreasing_schedule_ending_at_zero():
    shifted = FlowMatchEulerDiscreteScheduler(shift=3.0)
    unshifted = FlowMatchEulerDiscreteScheduler(shift=1.0)
    sigmas = sampler_sigmas(shifted, num_steps=4, image_seq_len=4096)
    baseline = sampler_sigmas(unshifted, num_steps=4, image_seq_len=4096)

    assert len(sigmas) == 5 and sigmas[0] == pytest.approx(1.0) and sigmas[-1] == 0.0
    assert all(a > b for a, b in zip(sigmas[:-1], sigmas[1:], strict=True))
    assert all(
        s > b for s, b in zip(sigmas[1:-1], baseline[1:-1], strict=True)
    )  # shift > 1 raises sigmas
    assert (
        getattr(shifted, "num_inference_steps", None) is None
    )  # the pipeline's scheduler is not modified


def test_sampler_sigmas_flux_style_linear_schedule():
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    sigmas = sampler_sigmas(scheduler, 4, 4096, sigmas=[1.0, 0.75, 0.5, 0.25])
    assert sigmas == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0], abs=1e-4)


def test_conditioning_for_cycles_over_prompts():
    embeds = torch.arange(3.0).view(3, 1, 1)
    setup = TextToImageSetup(
        family="sd3",
        conditioning={"encoder_hidden_states": embeds, "txt_ids": torch.zeros(1, 3)},
        noise=torch.zeros(3, 1),
        sigmas=[1.0, 0.0],
        decode=lambda x: x,
    )
    selected = setup.conditioning_for(torch.tensor([2, 3, 4]))
    assert selected["encoder_hidden_states"].flatten().tolist() == [2.0, 0.0, 1.0]
    assert selected["txt_ids"].shape == (1, 3)
