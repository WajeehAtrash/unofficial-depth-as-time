"""Turn velocity readouts into endpoint predictions (Eq. 3 of the paper)."""

from __future__ import annotations

from torch import Tensor

from depth_as_time.task import TransportTask


def _expand(scalar: Tensor, like: Tensor) -> Tensor:
    """Reshape a ``(B,)`` tensor to ``(B, 1, ..., 1)`` so it broadcasts against ``like``.

    ``like`` has shape ``(B, *S)``. The result also broadcasts against a tensor with an extra
    leading layer dimension, ``(L, B, *S)``.
    """
    return scalar.view(-1, *([1] * (like.ndim - 1)))


def endpoint_from_velocity(z: Tensor, velocity: Tensor, task: TransportTask) -> Tensor:
    """Layerwise endpoint probe ``x_r = z_t - (t - r) * v``.

    For an instantaneous velocity this is one Euler step. For an interval-averaged velocity
    (MeanFlow) it is the exact finite-interval parameterization.

    Args:
        z: Network input ``z_t`` with shape ``(B, *S)``.
        velocity: Velocity readout with shape ``(B, *S)`` or ``(L, B, *S)``.
        task: Transport task providing ``t`` and ``r``.

    Returns:
        Endpoint prediction with the same shape as ``velocity``.
    """
    return z - _expand(task.interval, z) * velocity


def clean_from_velocity(z: Tensor, velocity: Tensor, task: TransportTask) -> Tensor:
    """Diagnostic clean-image estimate ``x_0 = z_t - t * v`` (endpoint probe with ``r = 0``)."""
    return z - _expand(task.t, z) * velocity
