"""Shared test fixtures: a tiny residual network that satisfies the LayerwiseModel interface."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from depth_as_time.models.base import LayerwiseModel
from depth_as_time.task import TransportTask


class TinyResidualNet(LayerwiseModel):
    """Residual MLP with a time-conditioned head, used only for testing."""

    def __init__(self, data_dim: int = 4, width: int = 16, depth: int = 5) -> None:
        super().__init__()
        self.embed = nn.Linear(data_dim, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
            for _ in range(depth)
        )
        self.time_proj = nn.Linear(2, width)
        self.head = nn.Linear(width, data_dim)

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    def hidden_states(
        self, z: Tensor, task: TransportTask, label: Tensor | None = None
    ) -> list[Tensor]:
        cond = self.time_proj(torch.stack([task.t, task.r], dim=-1))
        h = self.embed(z)
        states = []
        for block in self.blocks:
            h = h + block(h + cond)
            states.append(h)
        return states

    def readout_tokens(self, h: Tensor, task: TransportTask, label: Tensor | None = None) -> Tensor:
        return self.head(h)


@pytest.fixture
def tiny_model() -> TinyResidualNet:
    torch.manual_seed(0)
    return TinyResidualNet().eval()
