"""Run the depth-as-time analyses on your own network.

The network must implement ``depth_as_time.LayerwiseModel``. Provide a factory function that
returns it, already loaded and in eval mode:

    # my_models.py
    def build_meanflow(checkpoint: str) -> LayerwiseModel: ...

    python scripts/probe_custom.py --factory my_models:build_meanflow \
        --factory-kwargs '{"checkpoint": "meanflow_b2.pt"}' \
        --input-shape 3 256 256 --num-classes 1000 --decode pixel \
        --sampler-times 1.0 0.5 0.0 --output-dir out/meanflow
"""

from __future__ import annotations

import argparse

import matplotlib
import torch

matplotlib.use("Agg")

from depth_as_time.cli import (  # noqa: E402
    add_report_arguments,
    load_factory,
    parse_dtype,
    parse_json_object,
    report_config_from_args,
)
from depth_as_time.report import generate_report, pixel_decoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--factory", required=True, help="'package.module:function' returning a LayerwiseModel."
    )
    parser.add_argument("--factory-kwargs", default="{}", help="JSON object passed to the factory.")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        required=True,
        help="Shape of one input, e.g. 3 256 256.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Number of samples probed.")
    parser.add_argument(
        "--num-classes", type=int, default=0, help="Classes for conditional models, 0 if none."
    )
    parser.add_argument(
        "--sampler-times",
        type=float,
        nargs="+",
        default=None,
        help="Decreasing times, e.g. 1.0 0.5 0.0.",
    )
    parser.add_argument(
        "--decode",
        choices=["none", "pixel"],
        default="none",
        help="'pixel' shows outputs as RGB images assuming values in [-1, 1].",
    )
    add_report_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = parse_dtype(args.dtype)
    factory = load_factory(args.factory)
    model = factory(**parse_json_object(args.factory_kwargs)).to(args.device, dtype).eval()

    generator = torch.Generator().manual_seed(args.seed)
    z = torch.randn(args.batch_size, *args.input_shape, generator=generator).to(args.device, dtype)
    label = None
    if args.num_classes > 0:
        label = torch.randint(0, args.num_classes, (args.batch_size,), generator=generator).to(
            args.device
        )

    config = report_config_from_args(args, args.sampler_times)
    generate_report(
        model,
        z,
        config,
        label=label,
        num_classes=args.num_classes,
        decode=pixel_decoder if args.decode == "pixel" else None,
    )
    print(f"Wrote results to {config.output_dir}")


if __name__ == "__main__":
    main()
