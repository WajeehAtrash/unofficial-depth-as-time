"""Transport task description shared by models, probes and samplers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TransportTask:
    """The transport ``t -> r`` a network is asked to solve for each batch element.

    Time follows the paper convention: ``t = 0`` is data and ``t = 1`` is noise, so a
    valid task has ``0 <= r <= t <= 1``. Step-conditioned models (Shortcut) receive a step
    size ``d = t - r``, which is available as :attr:`interval`.

    Attributes:
        t: Current noise level, shape ``(B,)``.
        r: Requested endpoint time, shape ``(B,)``.
    """

    t: Tensor
    r: Tensor

    def __post_init__(self) -> None:
        """Check that ``t`` and ``r`` are matching 1D tensors."""
        if self.t.shape != self.r.shape or self.t.ndim != 1:
            raise ValueError(
                f"t and r must be 1D tensors of equal shape, got {tuple(self.t.shape)} "
                f"and {tuple(self.r.shape)}"
            )

    @property
    def batch_size(self) -> int:
        """Number of batch elements the task describes."""
        return self.t.shape[0]

    @property
    def interval(self) -> Tensor:
        """Length of the transport interval, ``t - r``."""
        return self.t - self.r

    @classmethod
    def constant(
        cls,
        t: float,
        r: float,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> TransportTask:
        """Build a task with the same ``(t, r)`` for every batch element."""
        if not 0.0 <= r <= t <= 1.0:
            raise ValueError(f"Expected 0 <= r <= t <= 1, got t={t}, r={r}")
        return cls(
            t=torch.full((batch_size,), t, device=device, dtype=dtype),
            r=torch.full((batch_size,), r, device=device, dtype=dtype),
        )

    @classmethod
    def one_step(
        cls,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> TransportTask:
        """Full noise-to-data transport, ``t = 1`` and ``r = 0``."""
        return cls.constant(1.0, 0.0, batch_size, device=device, dtype=dtype)

    def to(self, device: torch.device | str) -> TransportTask:
        """Move the task tensors to ``device``."""
        return TransportTask(t=self.t.to(device), r=self.r.to(device))
