"""Quantities used to read the layerwise decodes: normalized distance and power spectra."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def layerwise_distances(predictions: Tensor, reference: Tensor) -> Tensor:
    """L2 distance of every layer's prediction to ``reference``.

    Args:
        predictions: Layerwise predictions of shape ``(L, B, *S)``.
        reference: Target of shape ``(B, *S)``.

    Returns:
        Distances of shape ``(L, B)``.
    """
    return (predictions - reference.unsqueeze(0)).flatten(2).norm(dim=-1)


def _mean_over_valid(normalized: Tensor, valid: Tensor) -> Tensor:
    """Average ``(L, B)`` curves over the trajectories flagged valid; NaN if none are."""
    if not valid.any():
        return torch.full((normalized.shape[0],), float("nan"), device=normalized.device)
    return normalized[:, valid].mean(dim=1)


def normalized_remaining_curves(
    blocks: Sequence[Tensor], reference: Tensor | None = None
) -> list[Tensor]:
    """Normalized denoising remaining ``rho(i)`` of the paper (Eq. 4).

    ``rho(i) = d(x^i, x^L) / max_j d(x^j, x^L)`` with ``d`` the L2 distance, normalized for each
    trajectory separately and then averaged over the batch. Trajectories whose distances are
    all zero are excluded.

    Args:
        blocks: One tensor ``(L, B, *S)`` of layerwise predictions per sampler step. A single
            one-step evaluation is a list with one element.
        reference: Optional target ``(B, *S)`` shared by all blocks, for example the final
            sample of a multi-step run. In that case the maximum is taken over all blocks,
            so the curves of different steps are comparable. When ``None`` each block is
            compared with its own last layer and normalized on its own.

    Returns:
        One curve of shape ``(L,)`` per block.
    """
    if reference is None:
        curves = []
        for block in blocks:
            distances = layerwise_distances(block, block[-1])
            maxima = distances.amax(dim=0)
            valid = maxima > 0
            normalized = distances / maxima.clamp_min(torch.finfo(distances.dtype).tiny)
            curves.append(_mean_over_valid(normalized, valid))
        return curves

    all_distances = [layerwise_distances(block, reference) for block in blocks]
    maxima = torch.stack(all_distances).amax(dim=(0, 1))
    valid = maxima > 0
    safe_maxima = maxima.clamp_min(torch.finfo(maxima.dtype).tiny)
    return [_mean_over_valid(d / safe_maxima, valid) for d in all_distances]


def normalized_remaining(predictions: Tensor, reference: Tensor | None = None) -> Tensor:
    """``rho(i)`` for a single block of layerwise predictions ``(L, B, *S)``."""
    return normalized_remaining_curves([predictions], reference)[0]


def radial_power_spectrum(images: Tensor) -> Tensor:
    """Radially averaged power spectrum ``P(kappa) = mean |F[x](omega)|^2`` for ``|omega| = kappa``.

    Args:
        images: Tensor of shape ``(*, C, H, W)``.

    Returns:
        Tensor of shape ``(*, K)`` with ``K = min(H, W) // 2 + 1`` radial bins, averaged over
        channels. Average over the leading dimensions to get a batch spectrum.
    """
    *lead, channels, height, width = images.shape
    flat = images.reshape(-1, channels, height, width).float()
    power = torch.fft.fftshift(torch.fft.fft2(flat, norm="ortho"), dim=(-2, -1)).abs().pow(2)
    power = power.mean(dim=1)  # (N, H, W)

    ys = torch.arange(height, device=images.device) - height // 2
    xs = torch.arange(width, device=images.device) - width // 2
    radius = torch.sqrt(ys[:, None].float() ** 2 + xs[None, :].float() ** 2).round().long()
    num_bins = min(height, width) // 2 + 1
    in_range = radius < num_bins

    counts = torch.bincount(radius[in_range], minlength=num_bins).clamp_min(1)
    spectra = torch.stack(
        [torch.bincount(radius[in_range], weights=p[in_range], minlength=num_bins) for p in power]
    )
    return (spectra / counts).reshape(*lead, num_bins)
