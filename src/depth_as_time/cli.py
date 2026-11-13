"""Command line helpers shared by the scripts."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from depth_as_time.report import ReportConfig

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options that map onto :class:`ReportConfig` plus device and dtype."""
    group = parser.add_argument_group("analysis")
    group.add_argument(
        "--output-dir", type=Path, required=True, help="Where figures and summary.json go."
    )
    group.add_argument(
        "--mixing-trajectories", type=int, default=256, help="0 skips the mixing fit."
    )
    group.add_argument("--mixing-batch-size", type=int, default=2)
    group.add_argument(
        "--no-controls", action="store_true", help="Skip the ridge and random probes."
    )
    group.add_argument("--decode-layers", type=int, default=12, help="Layers shown in image grids.")
    group.add_argument(
        "--visual-samples", type=int, default=2, help="Samples shown in image grids."
    )
    group.add_argument("--seed", type=int, default=0)
    group.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    group.add_argument("--dtype", choices=sorted(_DTYPES), default="float32")


def parse_dtype(name: str) -> torch.dtype:
    """Translate a dtype name from the command line."""
    return _DTYPES[name]


def report_config_from_args(
    args: argparse.Namespace, sampler_times: Sequence[float] | None
) -> ReportConfig:
    """Build a :class:`ReportConfig` from parsed arguments."""
    return ReportConfig(
        output_dir=args.output_dir,
        sampler_times=sampler_times,
        mixing_trajectories=args.mixing_trajectories,
        mixing_batch_size=args.mixing_batch_size,
        run_controls=not args.no_controls,
        num_decode_layers=args.decode_layers,
        num_visual_samples=args.visual_samples,
        seed=args.seed,
    )


def load_factory(spec: str) -> Any:
    """Import ``package.module:function`` and return the callable."""
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"Expected 'package.module:function', got {spec!r}")
    return getattr(importlib.import_module(module_name), attribute)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object given on the command line."""
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object such as \'{"checkpoint": "model.pt"}\'')
    return value
