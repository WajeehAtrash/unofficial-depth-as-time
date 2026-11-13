"""Readouts applied to intermediate hidden states.

The default readout is the model's own head (a generative logit lens). Two controls from
Appendix B.1 of the paper check that the layerwise pattern is not an artifact of that head:
a ridge-regression linear probe fit only on the last layer, and a random linear map with
matched scale.
"""

from __future__ import annotations

from collections.abc import Callable
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
        # The map is fit in float32, so apply it in float32 too (bf16 models would otherwise add
        # rounding noise to every layer's readout).
        features = F.layer_norm(h.float(), h.shape[-1:])
        return self.model.tokens_to_output(features @ self.weight.float())


@torch.no_grad()
def fit_ridge_readout(
    model: LayerwiseModel,
    z: Tensor,
    task: TransportTask,
    label: Tensor | None = None,
    ridge: float = 1e-3,
    batch_size: int | None = None,
    on_batch: Callable[[Tensor], None] | None = None,
) -> LinearTokenReadout:
    """Fit ``V`` so that ``V(LayerNorm(h_L)) ~ W_out(h_L)`` by closed-form ridge regression.

    The probe sees only the last hidden state, never an intermediate layer. The normal
    equations are accumulated batch by batch, so large models can be fitted with
    ``batch_size=1``.

    Args:
        model: Network to probe.
        z: Fitting inputs of shape ``(B, *S)``.
        task: Transport task for the fitting inputs.
        label: Optional labels.
        ridge: Regularization strength relative to the mean diagonal of the Gram matrix.
        batch_size: Inputs per network call. ``None`` uses the whole batch at once.
        on_batch: Optional callback called with the indices of each batch before the network
            call, for example to set the matching text conditioning.
    """
    size = z.shape[0] if batch_size is None else batch_size
    gram = xty = None
    for start in range(0, z.shape[0], size):
        idx = torch.arange(start, min(start + size, z.shape[0]))
        if on_batch is not None:
            on_batch(idx)
        sub_task = TransportTask(t=task.t[idx.to(task.t.device)], r=task.r[idx.to(task.r.device)])
        sub_label = None if label is None else label[idx.to(label.device)]
        h_last = model.hidden_states(z[idx.to(z.device)], sub_task, sub_label)[-1]
        targets = model.readout_tokens(h_last, sub_task, sub_label)
        x = F.layer_norm(h_last.float(), h_last.shape[-1:]).reshape(-1, h_last.shape[-1]).double()
        y = targets.reshape(-1, targets.shape[-1]).double()
        gram = x.T @ x if gram is None else gram + x.T @ x
        xty = x.T @ y if xty is None else xty + x.T @ y
    gram += (
        ridge
        * gram.diagonal().mean()
        * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    )
    weight = torch.linalg.solve(gram, xty)
    return LinearTokenReadout(model, weight.float())


def make_random_readout(
    reference: LinearTokenReadout, generator: torch.Generator | None = None
) -> LinearTokenReadout:
    """Random Gaussian map with the same shape and entrywise std as ``reference.weight``."""
    weight = reference.weight
    noise = torch.randn(weight.shape, generator=generator, dtype=weight.dtype)
    random_weight = noise.to(weight.device) * weight.std()
    return LinearTokenReadout(reference.model, random_weight)
