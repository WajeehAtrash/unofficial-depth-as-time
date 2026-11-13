"""Matplotlib figures mirroring the paper's analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch import Tensor

from depth_as_time.mixing import MixingResult


def plot_remaining_curves(curves: Mapping[str, Tensor], title: str = "") -> Figure:
    """Normalized denoising remaining ``rho(i)`` against normalized depth ``i / L`` (Figure 2)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, curve in curves.items():
        depth = torch.arange(1, len(curve) + 1) / len(curve)
        ax.plot(depth, curve.cpu(), marker="o", markersize=3, label=name)
    ax.set_xlabel("normalized depth i / L")
    ax.set_ylabel("normalized denoising remaining")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    if len(curves) > 1:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_multi_step_remaining(curves: Mapping[str, Sequence[Tensor]], title: str = "") -> Figure:
    """Per-step ``rho(i)`` laid end to end, with dashed lines at sampler steps (Figures 6, 7a)."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, steps in curves.items():
        num_layers = len(steps[0])
        values = torch.cat([c.cpu() for c in steps])
        ax.plot(
            torch.arange(len(values)) / num_layers, values, marker="o", markersize=2, label=name
        )
    for boundary in range(1, len(next(iter(curves.values())))):
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("sampler step (depth within each step)")
    ax.set_ylabel("normalized distance to final sample")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_mixing_schedule(result: MixingResult, title: str = "") -> Figure:
    """Recovered ``alpha_i`` and ``sigma_i`` per block (Figures 4 and 9)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    blocks = range(len(result.alpha))
    ax.plot(blocks, result.alpha, marker="o", markersize=3, label="alpha (final prediction)")
    ax.plot(blocks, result.sigma, marker="o", markersize=3, label="sigma (input noise)")
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.set_xlabel("block")
    ax.set_ylabel("recovered coefficient")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_held_out_residual(results: Mapping[str, MixingResult], title: str = "") -> Figure:
    """Held-out reconstruction error ``R_i`` of the two-component fit per block (Figure 3)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, result in results.items():
        ax.plot(
            range(len(result.held_out_r)), result.held_out_r, marker="o", markersize=3, label=name
        )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("block")
    ax.set_ylabel("held-out R_i")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_power_spectra(spectra: Tensor, labels: Sequence[str], title: str = "") -> Figure:
    """Radially averaged power spectra, one line per layer (Figure 18, bottom right)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    cmap = plt.get_cmap("viridis")
    for i, (spectrum, label) in enumerate(zip(spectra, labels, strict=True)):
        color = cmap(i / max(len(labels) - 1, 1))
        ax.loglog(range(1, len(spectrum)), spectrum[1:].cpu(), color=color, label=label)
    ax.set_xlabel("radial frequency")
    ax.set_ylabel("power")
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def plot_decode_grid(
    images: Tensor, row_labels: Sequence[str], thumbnail_size: int = 192, title: str = ""
) -> Figure:
    """Grid of decoded images, one row per layer and one column per sample (Figures 2, 5, 7).

    Args:
        images: RGB images in ``[0, 1]`` with shape ``(rows, cols, 3, H, W)``.
        row_labels: One label per row.
        thumbnail_size: Side length each image is resized to before plotting.
        title: Figure title.
    """
    rows, cols = images.shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols + 0.8, 1.6 * rows), squeeze=False)
    for r in range(rows):
        thumbs = F.interpolate(
            images[r], size=(thumbnail_size, thumbnail_size), mode="bilinear", antialias=True
        )
        for c in range(cols):
            ax = axes[r][c]
            ax.imshow(thumbs[c].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r][0].set_ylabel(row_labels[r], rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
