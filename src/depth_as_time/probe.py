"""The layerwise probe: decode every block with the model's own output head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from depth_as_time.decoding import clean_from_velocity, endpoint_from_velocity
from depth_as_time.models.base import LayerwiseModel
from depth_as_time.readouts import ModelReadout, Readout
from depth_as_time.task import TransportTask


@dataclass(frozen=True)
class ProbeResult:
    """Layerwise velocities for one network evaluation and their decoded predictions.

    Attributes:
        z: Network input ``z_t``, shape ``(B, *S)``.
        task: Transport task ``t -> r`` the network was asked to solve.
        velocities: ``v_i = W_out(h_i)`` for ``i = 1..L``, shape ``(L, B, *S)``.
    """

    z: Tensor
    task: TransportTask
    velocities: Tensor

    @property
    def num_layers(self) -> int:
        """Number of probed layers ``L``."""
        return self.velocities.shape[0]

    @property
    def endpoints(self) -> Tensor:
        """Layerwise endpoint predictions ``x_r^i = z_t - (t - r) v_i`` (Eq. 3)."""
        return endpoint_from_velocity(self.z, self.velocities, self.task)

    @property
    def clean(self) -> Tensor:
        """Layerwise clean-image predictions ``x_0^i = z_t - t v_i``."""
        return clean_from_velocity(self.z, self.velocities, self.task)

    @property
    def next_state(self) -> Tensor:
        """The model's ordinary update, ``z_t - (t - r) v_L``, used to advance a sampler."""
        return self.endpoints[-1]


class LayerwiseProbe:
    """Apply a readout to every hidden state without running the remaining blocks.

    Args:
        model: The one-step generator. Put it in ``eval()`` mode before probing.
        readout: Readout used on each layer. Defaults to the model's own head.
    """

    def __init__(self, model: LayerwiseModel, readout: Readout | None = None) -> None:
        self.model = model
        self.readout = readout if readout is not None else ModelReadout(model)

    @torch.no_grad()
    def run(self, z: Tensor, task: TransportTask, label: Tensor | None = None) -> ProbeResult:
        """Probe one network evaluation (Eq. 2).

        Args:
            z: Network input ``z_t`` of shape ``(B, *S)``.
            task: Transport task ``t -> r``.
            label: Optional class labels of shape ``(B,)``.
        """
        hidden = self.model.hidden_states(z, task, label)
        velocities = torch.stack([self.readout(h, task, label) for h in hidden])
        if velocities.dtype in (torch.float16, torch.bfloat16):
            velocities = velocities.float()  # decode and accumulate statistics in float32
        return ProbeResult(z=z, task=task, velocities=velocities)
