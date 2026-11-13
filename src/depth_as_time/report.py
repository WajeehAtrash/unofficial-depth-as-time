"""Run every analysis of the paper on one model and write figures plus a JSON summary."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from depth_as_time import plotting
from depth_as_time.metrics import normalized_remaining, radial_power_spectrum
from depth_as_time.mixing import MixingResult, run_mixing_analysis
from depth_as_time.models.base import LayerwiseModel
from depth_as_time.probe import LayerwiseProbe, ProbeResult
from depth_as_time.readouts import fit_ridge_readout, make_random_readout
from depth_as_time.task import TransportTask
from depth_as_time.trajectory import multi_step_remaining, probe_sampling_trajectory

Decoder = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class ReportConfig:
    """Options for :func:`generate_report`.

    Attributes:
        output_dir: Directory for figures and ``summary.json``.
        sampler_times: Decreasing times of the multi-step run, e.g. ``(1.0, 0.5, 0.0)``.
            ``None`` skips the multi-step analysis.
        mixing_trajectories: Number of one-step trajectories for the mixing fit (paper: 256).
            ``0`` skips the mixing analysis.
        mixing_batch_size: Trajectories per network call during the mixing analysis.
        run_controls: Also decode with a ridge probe and a random probe (Appendix B.1).
        num_decode_layers: Number of evenly spaced layers shown in image grids.
        num_visual_samples: Number of samples shown in image grids.
        seed: Seed for the mixing analysis and the random probe.
    """

    output_dir: Path
    sampler_times: Sequence[float] | None = None
    mixing_trajectories: int = 256
    mixing_batch_size: int = 8
    run_controls: bool = True
    num_decode_layers: int = 12
    num_visual_samples: int = 2
    seed: int = 0


def select_layers(num_layers: int, count: int) -> list[int]:
    """Evenly spaced layer indices that always include the first and last layer."""
    count = max(1, min(count, num_layers))
    if count == 1:
        return [num_layers - 1]
    return sorted({round(i * (num_layers - 1) / (count - 1)) for i in range(count)})


def pixel_decoder(images: Tensor) -> Tensor:
    """Decoder for models that output pixels in ``[-1, 1]``."""
    return ((images.float() + 1.0) / 2.0).clamp(0.0, 1.0).cpu()


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        return _to_jsonable(value.tolist())
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _decode_layers(
    decode: Decoder, predictions: Tensor, layers: Sequence[int], num_samples: int
) -> Tensor:
    """Decode selected layers of ``(L, B, ...)`` predictions to images ``(rows, cols, 3, H, W)``."""
    chosen = predictions[list(layers), :num_samples]
    rows, cols = chosen.shape[:2]
    images = decode(chosen.reshape(rows * cols, *chosen.shape[2:]))
    return images.reshape(rows, cols, *images.shape[1:])


def _one_step_rho(
    model: LayerwiseModel, z: Tensor, label: Tensor | None, config: ReportConfig
) -> tuple[dict[str, Tensor], ProbeResult]:
    task = TransportTask.one_step(z.shape[0], device=z.device, dtype=torch.float32)
    result = LayerwiseProbe(model).run(z, task, label)
    curves = {"model readout": normalized_remaining(result.clean)}
    if config.run_controls:
        ridge = fit_ridge_readout(model, z, task, label)
        generator = torch.Generator().manual_seed(config.seed)
        for name, readout in (
            ("ridge probe", ridge),
            ("random probe", make_random_readout(ridge, generator)),
        ):
            probe = LayerwiseProbe(model, readout)
            curves[name] = normalized_remaining(probe.run(z, task, label).clean)
    return curves, result


def _mixing(
    model: LayerwiseModel,
    z: Tensor,
    num_classes: int,
    on_batch: Callable[[Tensor], None] | None,
    config: ReportConfig,
) -> MixingResult:
    return run_mixing_analysis(
        LayerwiseProbe(model),
        noise_shape=tuple(z.shape[1:]),
        num_trajectories=config.mixing_trajectories,
        batch_size=config.mixing_batch_size,
        num_classes=num_classes,
        seed=config.seed,
        device=z.device,
        dtype=z.dtype,
        on_batch=on_batch,
    )


def generate_report(
    model: LayerwiseModel,
    z: Tensor,
    config: ReportConfig,
    *,
    label: Tensor | None = None,
    num_classes: int = 0,
    decode: Decoder | None = None,
    on_mixing_batch: Callable[[Tensor], None] | None = None,
    restore_conditioning: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run all analyses and write the results to ``config.output_dir``.

    Args:
        model: The one-step generator in ``eval()`` mode, already conditioned on the prompts or
            labels that match the batch of ``z``.
        z: Initial noise ``z_1`` of shape ``(B, *S)``.
        config: Report options.
        label: Optional class labels matching ``z``.
        num_classes: If positive, labels for the mixing analysis are drawn from this many classes.
        decode: Maps predictions ``(B, ...)`` to RGB images in ``[0, 1]``. Image grids and
            power spectra are skipped when ``None``.
        on_mixing_batch: Called before every mixing batch with the trajectory indices, to set
            per-batch conditioning.
        restore_conditioning: Called after the mixing analysis to put the original
            conditioning back.

    Returns:
        The summary dictionary that is also written to ``summary.json``.
    """
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    layers = select_layers(model.num_layers, config.num_decode_layers)
    layer_labels = [f"block {i}" for i in layers]
    summary: dict[str, Any] = {"num_layers": model.num_layers}

    curves, one_step = _one_step_rho(model, z, label, config)
    summary["one_step_rho"] = curves
    _save(
        plotting.plot_remaining_curves(curves, "One-step depth as time"), out / "one_step_rho.png"
    )
    if decode is not None:
        grid = _decode_layers(decode, one_step.clean, layers, config.num_visual_samples)
        _save(
            plotting.plot_decode_grid(grid, layer_labels, title="clean decode"),
            out / "one_step_decodes.png",
        )

    if config.mixing_trajectories > 0:
        mixing = _mixing(model, z, num_classes, on_mixing_batch, config)
        if restore_conditioning is not None:
            restore_conditioning()
        summary["mixing"] = {
            "alpha": mixing.alpha,
            "sigma": mixing.sigma,
            "held_out_r": mixing.held_out_r,
            "mean_held_out_r": mixing.mean_held_out_r,
            "num_train": mixing.num_train,
            "num_test": mixing.num_test,
        }
        _save(
            plotting.plot_mixing_schedule(mixing, "Recovered mixing schedule"),
            out / "mixing_schedule.png",
        )
        _save(plotting.plot_held_out_residual({"model": mixing}), out / "mixing_residual.png")

    if config.sampler_times is not None:
        summary["multi_step"] = _multi_step(model, z, label, decode, layers, layer_labels, config)
    (out / "summary.json").write_text(json.dumps(_to_jsonable(summary), indent=2))
    return summary


def _multi_step(
    model: LayerwiseModel,
    z: Tensor,
    label: Tensor | None,
    decode: Decoder | None,
    layers: Sequence[int],
    layer_labels: Sequence[str],
    config: ReportConfig,
) -> dict[str, Any]:
    out = config.output_dir
    times = list(config.sampler_times)
    results = probe_sampling_trajectory(LayerwiseProbe(model), z, times, label)
    curves = multi_step_remaining(results)
    _save(
        plotting.plot_multi_step_remaining(
            {"endpoint x_r": curves["endpoint"], "clean x_0": curves["clean"]},
            "Multi-step depth as time",
        ),
        out / "multi_step_rho.png",
    )
    if decode is not None:
        for k, result in enumerate(results, start=1):
            for kind, predictions in (("endpoint", result.endpoints), ("clean", result.clean)):
                grid = _decode_layers(decode, predictions, layers, config.num_visual_samples)
                title = f"step {k}: t={times[k - 1]:.3f} to r={times[k]:.3f}, {kind} decode"
                _save(
                    plotting.plot_decode_grid(grid, layer_labels, title=title),
                    out / f"step{k}_{kind}_decodes.png",
                )
        first = _decode_layers(decode, results[0].endpoints, layers, config.num_visual_samples)
        spectra = radial_power_spectrum(first.flatten(0, 1))
        spectra = spectra.reshape(len(layers), first.shape[1], -1).mean(dim=1)
        _save(
            plotting.plot_power_spectra(spectra, layer_labels, "Step 1 endpoint spectra"),
            out / "step1_power_spectra.png",
        )
    return {
        "times": times,
        "endpoint_rho": curves["endpoint"],
        "clean_rho": curves["clean"],
    }
