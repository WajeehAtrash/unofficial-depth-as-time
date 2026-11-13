"""Probe a multi-step sampling trajectory, one network evaluation at a time."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from depth_as_time.metrics import normalized_remaining_curves
from depth_as_time.probe import LayerwiseProbe, ProbeResult
from depth_as_time.task import TransportTask


def probe_sampling_trajectory(
    probe: LayerwiseProbe,
    z_init: Tensor,
    times: Sequence[float],
    label: Tensor | None = None,
) -> list[ProbeResult]:
    """Probe every subtask ``t_k -> t_{k+1}`` of a sampler (Algorithm E.1 of the paper).

    At each step the network is evaluated with the interval ``(t_k, t_{k+1})`` and every block
    is decoded. The sampler then advances with the model's normal output ``v_L``, so probing
    never changes the trajectory.

    For distilled models conditioned only on ``t`` the interval is implied by the sampler.
    For MeanFlow it is the explicit conditioning ``(t, r)``. For Shortcut models the
    adapter should convert ``task.interval`` to the step size ``d``.

    Args:
        probe: Layerwise probe.
        z_init: Initial noise ``z_1`` of shape ``(B, *S)``.
        times: Decreasing sampler times, e.g. ``[1.0, 0.5, 0.0]`` for two steps.
        label: Optional class labels.

    Returns:
        One :class:`ProbeResult` per sampler step.
    """
    if len(times) < 2:
        raise ValueError("times must contain at least a start and an end")
    if any(later > earlier for earlier, later in zip(times[:-1], times[1:], strict=True)):
        raise ValueError(f"times must be non-increasing, got {list(times)}")

    results: list[ProbeResult] = []
    z = z_init
    for t, r in zip(times[:-1], times[1:], strict=True):
        task = TransportTask.constant(t, r, z.shape[0], device=z.device, dtype=z.dtype)
        result = probe.run(z, task, label)
        results.append(result)
        z = result.next_state.to(z_init.dtype)
    return results


def multi_step_remaining(
    results: Sequence[ProbeResult], reference: Tensor | None = None
) -> dict[str, list[Tensor]]:
    """Normalized distance ``rho(i)`` of every sampler step (Figures 6 and 7a of the paper).

    Both the endpoint decodes ``x_r^i`` and the clean decodes ``x_0^i`` are compared with one
    reference, by default the final sample of the whole trajectory, and normalized by a shared
    maximum. Endpoint curves of a step with a noisy target therefore fall while the network
    reveals clean structure and rise again when it renoises toward the target
    (denoise-then-renoise). The paper defines Eq. 4 for one-step generation only, so using the
    final sample as the multi-step reference is an interpretation of its figures.

    Args:
        results: Output of :func:`probe_sampling_trajectory`.
        reference: Target of shape ``(B, *S)``. Defaults to the final sample.

    Returns:
        ``{"endpoint": [curve per step], "clean": [curve per step]}`` with curves of shape ``(L,)``.
    """
    if reference is None:
        reference = results[-1].next_state
    return {
        "endpoint": normalized_remaining_curves([r.endpoints for r in results], reference),
        "clean": normalized_remaining_curves([r.clean for r in results], reference),
    }
