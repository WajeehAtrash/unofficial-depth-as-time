import math

import pytest
import torch

from depth_as_time import (
    LayerwiseProbe,
    TransportTask,
    clean_from_velocity,
    endpoint_from_velocity,
    probe_sampling_trajectory,
    run_mixing_analysis,
)
from depth_as_time.metrics import (
    layerwise_distances,
    normalized_remaining,
    normalized_remaining_curves,
    normalized_remaining_from_distances,
    radial_power_spectrum,
)
from depth_as_time.mixing import MixingStatistics, evaluate_mixing
from depth_as_time.readouts import fit_ridge_readout, make_random_readout


def test_task_validation():
    with pytest.raises(ValueError):
        TransportTask.constant(0.3, 0.5, batch_size=2)
    with pytest.raises(ValueError):
        TransportTask(torch.zeros(2), torch.zeros(3))
    task = TransportTask.constant(0.8, 0.3, batch_size=2)
    assert torch.allclose(task.interval, torch.full((2,), 0.5))


def test_endpoint_and_clean_decoding():
    z = torch.randn(3, 4)
    v = torch.randn(6, 3, 4)
    task = TransportTask.constant(0.8, 0.3, batch_size=3)
    assert torch.allclose(endpoint_from_velocity(z, v, task), z - 0.5 * v)
    assert torch.allclose(clean_from_velocity(z, v, task), z - 0.8 * v)


def test_probe_last_layer_matches_model_output(tiny_model):
    z = torch.randn(3, 4)
    task = TransportTask.constant(0.7, 0.2, batch_size=3)
    result = LayerwiseProbe(tiny_model).run(z, task)
    assert result.velocities.shape == (tiny_model.num_layers, 3, 4)
    assert torch.allclose(result.velocities[-1], tiny_model(z, task), atol=1e-6)
    assert torch.allclose(result.next_state, z - 0.5 * tiny_model(z, task), atol=1e-6)


def test_rho_is_one_at_max_and_zero_at_end():
    layers = torch.stack([torch.full((2, 3), float(k)) for k in (4, 3, 2, 1, 0)])
    rho = normalized_remaining(layers)
    assert torch.allclose(rho, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))


def test_rho_excludes_all_zero_trajectories():
    layers = torch.zeros(4, 2, 3)
    layers[:2, 1] = 1.0  # sample 1 moves, sample 0 is constant
    rho = normalized_remaining(layers)
    assert torch.allclose(rho, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.isnan(normalized_remaining(torch.zeros(4, 2, 3))).all()


def test_rho_shared_reference_normalizes_across_steps():
    reference = torch.zeros(1, 2)
    step_a = torch.stack([torch.full((1, 2), 4.0), torch.zeros(1, 2)])
    step_b = torch.stack([torch.full((1, 2), 2.0), torch.zeros(1, 2)])
    curve_a, curve_b = normalized_remaining_curves([step_a, step_b], reference)
    assert torch.allclose(curve_a, torch.tensor([1.0, 0.0]))
    assert torch.allclose(curve_b, torch.tensor([0.5, 0.0]))


def test_mixing_recovers_known_coefficients():
    torch.manual_seed(0)
    num_layers, n, d = 4, 64, 50
    alpha = torch.tensor([0.2, 1.5, 2.18, 1.0])
    sigma = torch.tensor([-3.0, -6.04, -1.0, 0.0])
    noise = torch.randn(n, d)
    final = torch.randn(n, d)
    layers = alpha[:, None, None] * final + sigma[:, None, None] * noise
    layers[-1] = final

    train, test = MixingStatistics(num_layers), MixingStatistics(num_layers)
    train.update(layers[:, :40], final[:40], noise[:40])
    test.update(layers[:, 40:], final[40:], noise[40:])
    result = evaluate_mixing(train, test)

    assert torch.allclose(result.alpha[:3], alpha[:3], atol=1e-4)
    assert torch.allclose(result.sigma[:3], sigma[:3], atol=1e-4)
    assert (result.held_out_r[:3] < 1e-6).all()
    assert torch.isnan(result.held_out_r[3])
    assert (result.num_train, result.num_test) == (40, 24)


def test_streaming_statistics_match_single_update():
    torch.manual_seed(1)
    layers, final, noise = torch.randn(3, 10, 7), torch.randn(10, 7), torch.randn(10, 7)
    whole, parts = MixingStatistics(3), MixingStatistics(3)
    whole.update(layers, final, noise)
    parts.update(layers[:, :4], final[:4], noise[:4])
    parts.update(layers[:, 4:], final[4:], noise[4:])
    assert torch.allclose(whole.xf, parts.xf) and torch.allclose(whole.xe, parts.xe)
    assert math.isclose(whole.ee, parts.ee) and parts.count == 10


def test_run_mixing_analysis_split_and_reproducibility(tiny_model):
    probe = LayerwiseProbe(tiny_model)
    kwargs = dict(noise_shape=(4,), num_trajectories=100, batch_size=32, seed=3)
    first = run_mixing_analysis(probe, **kwargs)
    second = run_mixing_analysis(probe, **kwargs)
    assert (first.num_train, first.num_test) == (70, 30)
    assert torch.allclose(first.alpha, second.alpha)
    assert first.alpha.shape == (tiny_model.num_layers,)


def test_sampling_trajectory_follows_model_update(tiny_model):
    probe = LayerwiseProbe(tiny_model)
    z = torch.randn(2, 4)
    results = probe_sampling_trajectory(probe, z, [1.0, 0.5, 0.0])
    assert len(results) == 2

    manual = z
    for t, r in [(1.0, 0.5), (0.5, 0.0)]:
        task = TransportTask.constant(t, r, 2)
        manual = manual - (t - r) * tiny_model(manual, task)
    assert torch.allclose(results[-1].next_state, manual, atol=1e-5)
    with pytest.raises(ValueError):
        probe_sampling_trajectory(probe, z, [0.5, 1.0])


def test_ridge_readout_recovers_a_linear_head_and_random_matches_scale(tiny_model):
    class LinearHeadNet(type(tiny_model)):
        """Head is LayerNorm followed by a bias-free linear map, so a linear probe is exact."""

        def readout_tokens(self, h, task, label=None):
            return torch.nn.functional.layer_norm(h, h.shape[-1:]) @ self.head.weight.T

    model = LinearHeadNet().eval()
    z = torch.randn(256, 4)
    task = TransportTask.one_step(256)
    ridge = fit_ridge_readout(model, z, task, ridge=1e-8)

    h = model.hidden_states(z, task)[2]
    assert torch.allclose(ridge(h, task), model.readout(h, task), atol=1e-3)

    rand = make_random_readout(ridge, torch.Generator().manual_seed(0))
    assert rand.weight.shape == ridge.weight.shape
    assert math.isclose(rand.weight.std().item(), ridge.weight.std().item(), rel_tol=0.35)
    assert not torch.allclose(rand(h, task), ridge(h, task))


def test_rho_from_distances_matches_rho_from_predictions():
    torch.manual_seed(0)
    steps = [torch.randn(5, 3, 4) for _ in range(2)]
    reference = torch.randn(3, 4)
    expected = normalized_remaining_curves(steps, reference)
    got = normalized_remaining_from_distances([layerwise_distances(s, reference) for s in steps])
    for e, g in zip(expected, got, strict=True):
        assert torch.allclose(e, g)


def test_ridge_readout_streaming_matches_full_batch(tiny_model):
    torch.manual_seed(1)
    z = torch.randn(12, 4)
    task = TransportTask.one_step(12)
    seen = []
    full = fit_ridge_readout(tiny_model, z, task).weight
    streamed = fit_ridge_readout(
        tiny_model, z, task, batch_size=5, on_batch=lambda idx: seen.append(idx.tolist())
    ).weight
    assert torch.allclose(full, streamed, atol=1e-5)
    assert seen == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]


def test_radial_power_spectrum_of_constant_image_is_dc_only():
    images = torch.ones(2, 3, 16, 16)
    spectrum = radial_power_spectrum(images)
    assert spectrum.shape == (2, 9)
    assert (spectrum[:, 0] > 0).all()
    assert torch.allclose(spectrum[:, 1:], torch.zeros(2, 8), atol=1e-6)


def test_multi_step_remaining_uses_final_sample_as_shared_reference(tiny_model):
    from depth_as_time import multi_step_remaining

    probe = LayerwiseProbe(tiny_model)
    results = probe_sampling_trajectory(probe, torch.randn(3, 4), [1.0, 0.5, 0.0])
    curves = multi_step_remaining(results)
    assert len(curves["endpoint"]) == len(curves["clean"]) == 2
    assert curves["endpoint"][0].shape == (tiny_model.num_layers,)
    # The last step ends at the reference, so its final endpoint distance is zero.
    assert curves["endpoint"][-1][-1].item() == pytest.approx(0.0, abs=1e-6)
    # Every curve is normalized by a shared maximum, so no value exceeds one.
    assert max(c.max().item() for c in curves["clean"] + curves["endpoint"]) <= 1.0 + 1e-6


def test_on_batch_receives_global_indices(tiny_model):
    seen = []
    run_mixing_analysis(
        LayerwiseProbe(tiny_model),
        noise_shape=(4,),
        num_trajectories=10,
        batch_size=4,
        on_batch=lambda idx: seen.append(idx.tolist()),
    )
    assert seen == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
