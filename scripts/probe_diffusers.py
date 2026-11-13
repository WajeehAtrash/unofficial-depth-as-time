"""Run the depth-as-time analyses on a diffusers text-to-image model.

Examples:
    python scripts/probe_diffusers.py --model flux-schnell --output-dir out/flux \
        --prompt "a photo of a red fox in snow" --dtype bfloat16
    python scripts/probe_diffusers.py --model stabilityai/stable-diffusion-3.5-large-turbo \
        --output-dir out/sd35 --prompt "a lighthouse at dusk" --dtype bfloat16 --num-steps 4
    # Reproduce the paper's FLUX analysis over the 38 single-stream blocks only:
    python scripts/probe_diffusers.py --model flux-schnell --output-dir out/flux_sb \
        --prompt "a lighthouse at dusk" --block-attrs single_transformer_blocks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")

from depth_as_time.cli import (  # noqa: E402
    add_report_arguments,
    parse_dtype,
    report_config_from_args,
)
from depth_as_time.models.diffusers_adapter import DiffusersTransformerAdapter  # noqa: E402
from depth_as_time.report import generate_report  # noqa: E402
from depth_as_time.text_to_image import (  # noqa: E402
    MODEL_PRESETS,
    load_pipeline,
    prepare_text_to_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            f"Preset ({', '.join(MODEL_PRESETS)}) or any Hugging Face repo id of a FLUX or SD3 "
            "pipeline."
        ),
    )
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument(
        "--prompt", action="append", help="Prompt. Repeat the flag for several prompts."
    )
    prompts.add_argument("--prompts-file", type=Path, help="Text file with one prompt per line.")
    parser.add_argument(
        "--num-steps", type=int, default=4, help="Sampler steps for the multi-step analysis."
    )
    parser.add_argument(
        "--one-step-only", action="store_true", help="Skip the multi-step analysis."
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Only used by FLUX dev.")
    parser.add_argument(
        "--block-attrs",
        nargs="+",
        default=None,
        help="Block lists to probe, e.g. single_transformer_blocks. Default: all blocks.",
    )
    add_report_arguments(parser)
    return parser.parse_args()


def read_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return args.prompt
    lines = args.prompts_file.read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


def main() -> None:
    args = parse_args()
    dtype = parse_dtype(args.dtype)
    prompts = read_prompts(args)

    pipe = load_pipeline(MODEL_PRESETS.get(args.model, args.model), dtype)
    setup = prepare_text_to_image(
        pipe,
        prompts,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
        max_sequence_length=args.max_sequence_length,
        guidance_scale=args.guidance_scale,
    )
    pipe.transformer.to(args.device).eval()

    adapter = DiffusersTransformerAdapter(pipe.transformer, block_attrs=args.block_attrs)
    adapter.set_conditioning(**setup.conditioning)
    config = report_config_from_args(args, None if args.one_step_only else setup.sigmas)

    with torch.no_grad():
        generate_report(
            adapter,
            setup.noise,
            config,
            decode=setup.decode,
            on_mixing_batch=lambda idx: adapter.set_conditioning(**setup.conditioning_for(idx)),
            restore_conditioning=lambda: adapter.set_conditioning(**setup.conditioning),
        )
    print(f"Wrote results to {config.output_dir}")


if __name__ == "__main__":
    main()
