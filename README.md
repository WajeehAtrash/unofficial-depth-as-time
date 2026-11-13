# Depth as Time in One-Step Generative Models (unofficial)

An unofficial PyTorch implementation of the analysis in
[Depth as Time in One-Step Generative Models](https://arxiv.org/abs/2610.03626)
(Asiimwe, Yang, Chun, Tureci, Russakovsky). The authors have not released code, so this
repository is rebuilt from the paper text and appendix.

The paper's observation: the denoising work that a multi-step sampler spreads over sampling
steps shows up across the layers of a single forward pass of a one-step generator. It is read
out with no training by applying the model's own output head to every intermediate hidden
state (a generative logit lens).

## Scope

Implemented:

| Paper section | What | Where |
|---|---|---|
| 3.1, Eq. 2 and 3 | Layerwise probe, velocity to endpoint decoding | `probe.py`, `decoding.py` |
| 3.2, Eq. 4 | Normalized denoising remaining `rho(i)` | `metrics.py` |
| 3.2, Eq. 5, App. B.2 | Depthwise mixing schedule `(alpha_i, sigma_i)` and held-out `R_i` | `mixing.py` |
| App. B.1 | Readout controls: ridge linear probe and random probe | `readouts.py` |
| 4.1, 4.2, App. E.1 | Per-sampler-step probing (distilled, MeanFlow, Shortcut), denoise-then-renoise | `trajectory.py` |
| App. E.1 | Radially averaged power spectrum | `metrics.py` |
| Figures 2 to 7, 9, 18 | Plots and decode grids | `plotting.py`, `report.py` |
| (adapter) | Hook-based support for diffusers FLUX and SD3 families | `models/diffusers_adapter.py`, `text_to_image.py` |

Not implemented, by request: the compression method of Section 5 (shared clock-conditioned
block, fixed-K and elastic training, FD-loss post-training) and no model training of any kind.
This repository only analyzes existing models.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

Tested with Python 3.12, torch 2.15, diffusers 0.42.1 and transformers 5.22.

Gated models (for example SD3.5) need an accepted license and a Hugging Face login
(`huggingface-cli login`) before the first download.

## Quick start: diffusers text-to-image models

```bash
# FLUX.1 schnell, all blocks (19 double-stream + 38 single-stream)
python scripts/probe_diffusers.py --model flux-schnell --dtype bfloat16 \
    --prompt "a photo of a red fox in snow" --output-dir out/flux

# FLUX.1 schnell, single-stream blocks only (the paper's L = 38 setting)
python scripts/probe_diffusers.py --model flux-schnell --dtype bfloat16 \
    --prompt "a photo of a red fox in snow" --block-attrs single_transformer_blocks \
    --output-dir out/flux_sb

# SD3.5 Large Turbo
python scripts/probe_diffusers.py --model sd35-large-turbo --dtype bfloat16 \
    --prompt "a lighthouse at dusk" --output-dir out/sd35

# Any other FLUX or SD3 pipeline by repository id, for example an SD3 Medium turbo model
python scripts/probe_diffusers.py --model <hf-repo-id> --dtype bfloat16 \
    --prompt "a lighthouse at dusk" --output-dir out/custom
```

Presets are `flux-schnell` (`black-forest-labs/FLUX.1-schnell`) and `sd35-large-turbo`
(`stabilityai/stable-diffusion-3.5-large-turbo`). Any repository id that loads as a diffusers
FLUX or SD3 pipeline works, because blocks are captured with forward hooks and nothing in the
transformer code is copied.

Useful options:

- `--num-steps 4` sampler steps of the multi-step analysis. The sigma schedule is read from the
  pipeline's own scheduler. `--one-step-only` skips this analysis.
- `--height/--width` default to 512 to keep memory low. All hidden states of all blocks are kept
  during a probe, so memory grows with resolution, prompts and block count.
- `--prompt` can be repeated, or use `--prompts-file`. The mixing fit cycles through the prompts.
- `--mixing-trajectories 256` follows the paper. `0` skips the fit.
- `--decode-layers`, `--visual-samples` control the image grids.

## Quick start: your own network

Implement `depth_as_time.LayerwiseModel` (three methods) and point the script at a factory:

```python
from depth_as_time import LayerwiseModel, TransportTask

class MyAdapter(LayerwiseModel):
    @property
    def num_layers(self) -> int: ...

    def hidden_states(self, z, task: TransportTask, label=None) -> list:
        """Return [h_1, ..., h_L], the residual stream after each block."""

    def readout_tokens(self, h, task: TransportTask, label=None):
        """Apply the output head W_out (norm, AdaLN, linear) to one hidden state."""

    def tokens_to_output(self, tokens):  # optional, for example unpatchify
        return tokens
```

```bash
python scripts/probe_custom.py --factory my_models:build_meanflow \
    --factory-kwargs '{"checkpoint": "meanflow_b2.pt"}' \
    --input-shape 3 256 256 --num-classes 1000 --decode pixel \
    --sampler-times 1.0 0.5 0.0 --output-dir out/meanflow
```

`TransportTask` carries `(t, r)` for the batch. Convention: data at `t = 0`, noise at `t = 1`,
linear path `x_t = (1 - t) x_0 + t eps`. Shortcut models receive a step size `d`, which is
`task.interval`. Models that use the opposite time convention must convert inside the adapter.

Python API:

```python
from depth_as_time import (
    LayerwiseProbe, TransportTask, run_mixing_analysis,
    probe_sampling_trajectory, multi_step_remaining,
)
from depth_as_time.metrics import normalized_remaining

probe = LayerwiseProbe(model)                       # model's own head as readout
result = probe.run(z, TransportTask.one_step(z.shape[0]), label)
result.clean                                        # x_0^i for every layer, shape (L, B, ...)
rho = normalized_remaining(result.clean)            # Eq. 4

mixing = run_mixing_analysis(probe, noise_shape=z.shape[1:], num_classes=1000)
mixing.alpha, mixing.sigma, mixing.held_out_r       # Eq. 5, per block

steps = probe_sampling_trajectory(probe, z, [1.0, 0.5, 0.0], label)
curves = multi_step_remaining(steps)                # denoise-then-renoise diagnostics
```

## Outputs

Each script writes to `--output-dir`:

- `summary.json`: `rho` curves, mixing coefficients, held-out `R_i`, multi-step curves.
- `one_step_rho.png`: `rho(i)` for the model readout, ridge probe and random probe.
- `one_step_decodes.png`: clean decodes of selected layers.
- `mixing_schedule.png`, `mixing_residual.png`: recovered `(alpha_i, sigma_i)` and held-out `R_i`.
- `multi_step_rho.png`, `stepK_{endpoint,clean}_decodes.png`, `step1_power_spectra.png`.

## Design notes and interpretation choices

The paper does not specify everything. These are the choices made here, so results can be read
correctly.

1. **Layers.** The probe decodes `h_1, ..., h_L`, the output of each block. The embedding
   output `h_0` is not decoded.
2. **Multi-step `rho`.** Eq. 4 is defined for one-step generation. For multi-step runs the
   paper's Figures 6 and 7a are reproduced by comparing each step's decodes with the final
   sample of the whole trajectory and normalizing with one shared maximum. This is an
   interpretation of the figures, not a formula given in the text. Pass `reference=` to
   `multi_step_remaining` to change it.
3. **Mixing fit.** Solved in closed form from streamed inner products, so memory does not grow
   with image size or trajectory count. The split is 70/30 by trajectory index (179/77 for
   256). For text-to-image models the prompt of each trajectory cycles through the prompt
   list, since the paper does not say how prompts were chosen.
4. **Ridge probe.** Fit on `LayerNorm(h_L)` only, without an intercept, with a ridge term
   relative to the mean diagonal of the Gram matrix. The random probe copies its scale.
5. **Precision.** Low precision velocities are upcast to float32 before decoding and before any
   statistic is accumulated, and sums run in float64.
6. **Text tokens.** For joint-attention transformers only the image token stream is decoded.
   Text conditioning is fixed per adapter (`adapter.set_conditioning`). A conditioning batch of
   size 1 is broadcast over the noise batch.
7. **Reading the decodes.** For a velocity model, `x_0^i = z - v_i` contains the input noise by
   construction, so part of the recovered `sigma_i` structure follows from the decoding rule
   itself. Compare against the Drifting-style negative control with care, and prefer
   conclusions that also hold for the ridge probe and the held-out `R_i`.

## Verification status

Unit tests cover the decoding rule, `rho`, the mixing fit (exact coefficient recovery on
synthetic data, streaming equals batch statistics), readout controls, trajectory bookkeeping,
and the diffusers adapter (the last-layer readout reproduces the real transformer output for
FLUX and SD3, and a multi-step run reproduces a manual sampler). The scripts were also run end
to end on the small `hf-internal-testing/tiny-flux-pipe` and `tiny-sd3-pipe` pipelines.

The full-size FLUX.1 schnell model and a MeanFlow SiT-B/4 checkpoint have also been run and
compared with the paper's figures and numbers. MeanFlow agrees to within about 1-2%, and FLUX
agrees to within 1-4 percentage points on the rho curves, with one gap in the size of the FLUX
noise coefficient. The comparison, the regenerated figures and a code review of the gap are in
[paper_reproduction/REPORT.md](paper_reproduction/REPORT.md). SD3.5 and other models have not
been run.

```bash
pytest
ruff check src tests scripts
```

## Layout

```
src/depth_as_time/
    task.py                  TransportTask (t, r)
    decoding.py              velocity to endpoint, Eq. 3
    probe.py                 LayerwiseProbe, ProbeResult
    readouts.py              model head, ridge probe, random probe
    metrics.py               rho(i), radial power spectrum
    mixing.py                mixing schedule fit and held-out error
    trajectory.py            multi-step probing, denoise-then-renoise curves
    models/base.py           LayerwiseModel interface
    models/diffusers_adapter.py  hook-based adapter for FLUX and SD3 transformers
    text_to_image.py         prompts, noise, sigmas and VAE decoding for pipelines
    plotting.py, report.py   figures and the full analysis report
    cli.py                   shared command line helpers
scripts/                     probe_diffusers.py, probe_custom.py
paper_reproduction/          reproduction of the paper's FLUX and MeanFlow results, see REPORT.md
tests/
```

## Citation

```bibtex
@article{asiimwe2026depth,
  title   = {Depth as Time in One-Step Generative Models},
  author  = {Asiimwe, Arnold Caleb and Yang, William and Chun, Sanghyuk and Tureci, Esin and Russakovsky, Olga},
  journal = {arXiv preprint arXiv:2610.03626},
  year    = {2026}
}
```

This repository is not affiliated with the authors.
