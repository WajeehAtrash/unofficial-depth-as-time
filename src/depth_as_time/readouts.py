"""Readouts applied to intermediate hidden states.

The default readout is the model's own head (a generative logit lens). Two controls from
Appendix B.1 of the paper check that the layerwise pattern is not an artifact of that head:
a ridge-regression linear probe fit only on the last layer, and a random linear map with
matched scale.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from depth_as_time.models.base import LayerwiseModel
from depth_as_time.task import TransportTask


class Readout(Protocol):
    """Maps a hidden state to a velocity with the shape of the model input."""

    def __call__(self, h: Tensor, task: TransportTask, label: Tensor | None = None) -> Tensor:
        """Decode ``h``."""


class ModelReadout:
    """The model's own output head ``W_out`` (Eq. 2)."""

    def __init__(self, model: LayerwiseModel) -> None:
        self.model = model

    def __call__(self, h: Tensor, task: TransportTask, label: Tensor | None = None) -> Tensor:
        """Apply the model head."""
        return self.model.readout(h, task, label)


class LinearTokenReadout:
    """``LayerNorm`` followed by a single linear map, applied token-wise.

    The map ignores the task and label, so it is the same for every layer.
    """

    def __init__(self, model: LayerwiseModel, weight: Tensor) -> None:
        self.model = model
        self.weight = weight

    def __call__(self, h: Tensor, task: TransportTask, label: Tensor | None = None) -> Tensor:
        """Apply LayerNorm and the linear map to ``h``."""
        features = F.layer_norm(h, h.shape[-1:])
        return self.model.tokens_to_output(features @ self.weight.to(features.dtype))


@torch.no_grad()
def fit_ridge_readout(
    model: LayerwiseModel,
    z: Tensor,
    task: TransportTask,
    label: Tensor | None = None,
    ridge: float = 1e-3,
) -> LinearTokenReadout:
    """Fit ``V`` so that ``V(LayerNorm(h_L)) ~ W_out(h_L)`` by closed-form ridge regression.

    The probe sees only the last hidden state, never an intermediate layer.

    Args:
        model: Network to probe.
        z: Fitting inputs of shape ``(B, *S)``.
        task: Transport task for the fitting inputs.
        label: Optional labels.
        ridge: Regularization strength relative to the mean diagonal of the Gram matrix.
    """
    h_last = model.hidden_states(z, task, label)[-1]
    targets = model.readout_tokens(h_last, task, label)
    features = F.layer_norm(h_last, h_last.shape[-1:])

    x = features.reshape(-1, features.shape[-1]).double()
    y = targets.reshape(-1, targets.shape[-1]).double()
    gram = x.T @ x
    gram += (
        ridge
        * gram.diagonal().mean()
        * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    )
    weight = torch.linalg.solve(gram, x.T @ y)
    return LinearTokenReadout(model, weight.float())


def make_random_readout(
    reference: LinearTokenReadout, generator: torch.Generator | None = None
) -> LinearTokenReadout:
    """Random Gaussian map with the same shape and entrywise std as ``reference.weight``."""
    weight = reference.weight
    noise = torch.randn(weight.shape, generator=generator, dtype=weight.dtype)
    random_weight = noise.to(weight.device) * weight.std()
    return LinearTokenReadout(reference.model, random_weight)
