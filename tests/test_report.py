import json

import torch

from depth_as_time.report import ReportConfig, generate_report, select_layers


def test_select_layers_includes_first_and_last():
    assert select_layers(57, 12)[0] == 0 and select_layers(57, 12)[-1] == 56
    assert select_layers(5, 12) == [0, 1, 2, 3, 4]
    assert select_layers(5, 1) == [4]


def test_generate_report_writes_summary_and_figures(tiny_model, tmp_path):
    config = ReportConfig(
        output_dir=tmp_path,
        sampler_times=(1.0, 0.5, 0.0),
        mixing_trajectories=40,
        mixing_batch_size=16,
    )
    summary = generate_report(tiny_model, torch.randn(8, 4), config)

    written = {p.name for p in tmp_path.iterdir()}
    assert {
        "summary.json",
        "one_step_rho.png",
        "mixing_schedule.png",
        "multi_step_rho.png",
    } <= written
    assert set(summary["one_step_rho"]) == {"model readout", "ridge probe", "random probe"}
    loaded = json.loads((tmp_path / "summary.json").read_text())
    assert len(loaded["mixing"]["alpha"]) == tiny_model.num_layers
    assert loaded["mixing"]["held_out_r"][-1] is None  # NaN is written as null
    assert len(loaded["multi_step"]["endpoint_rho"]) == 2


def test_generate_report_end_to_end_on_tiny_sd3(tmp_path):
    import pytest

    pytest.importorskip("diffusers")
    import torch.nn.functional as F

    from depth_as_time.models.diffusers_adapter import DiffusersTransformerAdapter
    from tests.test_diffusers_adapter import _tiny_sd3

    model, conditioning, z = _tiny_sd3()
    adapter = DiffusersTransformerAdapter(model)
    adapter.set_conditioning(**conditioning)

    def decode(latents):  # stand-in for the VAE: 4 latent channels to 3 RGB channels at 32x32
        rgb = latents.float()[:, :3].tanh() / 2 + 0.5
        return F.interpolate(rgb, size=(32, 32), mode="nearest")

    config = ReportConfig(
        output_dir=tmp_path,
        sampler_times=(1.0, 0.5, 0.0),
        mixing_trajectories=8,
        mixing_batch_size=4,
        num_decode_layers=3,
    )
    generate_report(
        adapter,
        z,
        config,
        decode=decode,
        on_mixing_batch=lambda idx: adapter.set_conditioning(
            **{
                **conditioning,
                "encoder_hidden_states": conditioning["encoder_hidden_states"][:1],
                "pooled_projections": conditioning["pooled_projections"][:1],
            }
        ),
    )

    written = {p.name for p in tmp_path.iterdir()}
    assert {
        "one_step_decodes.png",
        "step1_endpoint_decodes.png",
        "step2_clean_decodes.png",
        "step1_power_spectra.png",
        "mixing_residual.png",
    } <= written
