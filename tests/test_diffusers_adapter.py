import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from depth_as_time import LayerwiseProbe, TransportTask, probe_sampling_trajectory  # noqa: E402
from depth_as_time.models.diffusers_adapter import DiffusersTransformerAdapter  # noqa: E402

TEXT_TOKENS = 5


def _tiny_flux():
    from diffusers import FluxTransformer2DModel

    torch.manual_seed(0)
    model = FluxTransformer2DModel(
        patch_size=1,
        in_channels=8,
        num_layers=2,
        num_single_layers=3,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=24,
        pooled_projection_dim=12,
        axes_dims_rope=(4, 6, 6),
    ).eval()
    grid = 4
    batch = 2
    img_ids = torch.zeros(grid, grid, 3)
    img_ids[..., 1] = torch.arange(grid)[:, None]
    img_ids[..., 2] = torch.arange(grid)[None, :]
    conditioning = dict(
        encoder_hidden_states=torch.randn(batch, TEXT_TOKENS, 24),
        pooled_projections=torch.randn(batch, 12),
        txt_ids=torch.zeros(TEXT_TOKENS, 3),
        img_ids=img_ids.reshape(grid * grid, 3),
    )
    z = torch.randn(batch, grid * grid, 8)
    return model, conditioning, z


def _tiny_sd3():
    from diffusers import SD3Transformer2DModel

    torch.manual_seed(0)
    model = SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        num_layers=3,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=24,
        caption_projection_dim=16,
        pooled_projection_dim=12,
        out_channels=4,
        pos_embed_max_size=16,
    ).eval()
    conditioning = dict(
        encoder_hidden_states=torch.randn(2, TEXT_TOKENS, 24),
        pooled_projections=torch.randn(2, 12),
    )
    z = torch.randn(2, 4, 8, 8)
    return model, conditioning, z


def _reference_output(model, conditioning, z, task, timestep_scale):
    with torch.no_grad():
        return model(
            hidden_states=z, timestep=task.t * timestep_scale, return_dict=False, **conditioning
        )[0]


@pytest.mark.parametrize("builder", [_tiny_flux, _tiny_sd3], ids=["flux", "sd3"])
def test_last_layer_readout_matches_transformer_output(builder):
    model, conditioning, z = builder()
    adapter = DiffusersTransformerAdapter(model)
    adapter.set_conditioning(**conditioning)
    task = TransportTask.constant(0.6, 0.1, batch_size=z.shape[0])

    result = LayerwiseProbe(adapter).run(z, task)
    expected = _reference_output(model, conditioning, z, task, adapter.spec.timestep_scale)

    assert result.velocities.shape == (adapter.num_layers, *expected.shape)
    assert torch.allclose(result.velocities[-1], expected, atol=1e-5)
    assert not torch.allclose(result.velocities[0], result.velocities[-1])


def test_flux_layer_counts_and_block_subset():
    model, conditioning, z = _tiny_flux()
    assert DiffusersTransformerAdapter(model).num_layers == 2 + 3
    single_only = DiffusersTransformerAdapter(model, block_attrs=("single_transformer_blocks",))
    single_only.set_conditioning(**conditioning)
    states = single_only.hidden_states(z, TransportTask.one_step(2))
    assert len(states) == 3
    assert states[0].shape == (2, 16, model.inner_dim)  # image tokens only, no text tokens


def test_sd3_multi_step_trajectory_matches_manual_sampling():
    model, conditioning, z = _tiny_sd3()
    adapter = DiffusersTransformerAdapter(model)
    adapter.set_conditioning(**conditioning)
    results = probe_sampling_trajectory(LayerwiseProbe(adapter), z, [1.0, 0.5, 0.0])

    manual = z
    for t, r in [(1.0, 0.5), (0.5, 0.0)]:
        task = TransportTask.constant(t, r, z.shape[0])
        manual = manual - (t - r) * _reference_output(model, conditioning, manual, task, 1000.0)
    assert torch.allclose(results[-1].next_state, manual, atol=1e-4)


def test_close_removes_hooks():
    model, conditioning, z = _tiny_flux()
    adapter = DiffusersTransformerAdapter(model)
    adapter.close()
    adapter.set_conditioning(**conditioning)
    with pytest.raises(RuntimeError):
        adapter.hidden_states(z, TransportTask.one_step(2))


def test_single_prompt_is_broadcast_over_noise_batch():
    model, conditioning, _ = _tiny_flux()
    one_prompt = {
        **conditioning,
        "encoder_hidden_states": conditioning["encoder_hidden_states"][:1],
        "pooled_projections": conditioning["pooled_projections"][:1],
    }
    adapter = DiffusersTransformerAdapter(model)
    adapter.set_conditioning(**one_prompt)
    z = torch.randn(5, 16, 8)
    assert adapter.hidden_states(z, TransportTask.one_step(5))[-1].shape[0] == 5

    adapter.set_conditioning(**conditioning)  # batch size 2 does not match 5
    with pytest.raises(ValueError):
        adapter.hidden_states(z, TransportTask.one_step(5))
