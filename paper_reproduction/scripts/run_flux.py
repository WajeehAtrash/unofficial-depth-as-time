"""Run every FLUX.1-schnell analysis of the paper and save raw results to ``data/flux/``.

Hidden states of all 57 blocks at 1024x1024 take about 1.4 GB per sample, so everything is
streamed one or two samples at a time and only per-trajectory distances are kept.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from common import DATA, free_cuda, save_json
from depth_as_time import LayerwiseProbe, TransportTask, probe_sampling_trajectory
from depth_as_time.metrics import layerwise_distances
from depth_as_time.mixing import MixingStatistics, evaluate_mixing, run_mixing_analysis
from depth_as_time.models.diffusers_adapter import DiffusersTransformerAdapter
from depth_as_time.readouts import fit_ridge_readout, make_random_readout
from depth_as_time.text_to_image import load_pipeline, prepare_text_to_image

PROMPTS = [
    "a blue jay perched on a branch of pink cherry blossoms",
    "a photo of a red fox in snow",
    "a lighthouse at dusk on a rocky coast",
    "a bowl of ramen with a soft boiled egg, top view",
    "a golden retriever puppy running on grass",
    "a vintage red car parked on a cobblestone street",
    "a snowy mountain range under a clear blue sky",
    "a cup of cappuccino with latte art on a wooden table",
    "a giant panda eating bamboo in a forest",
    "an astronaut riding a horse on the moon",
    "a monarch butterfly on a purple flower",
    "a medieval castle on a hill at sunset",
    "a bowl of fresh strawberries and blueberries",
    "a tabby cat sleeping on a windowsill",
    "a futuristic city skyline at night with neon lights",
    "an old wooden boat on a calm lake in the morning mist",
    "a portrait of an elderly man with a white beard",
    "a field of sunflowers under a stormy sky",
    "a hot air balloon over a desert canyon",
    "a stack of pancakes with maple syrup and butter",
    "a green parrot sitting on a tree branch in the jungle",
    "a cozy living room with a fireplace and bookshelves",
    "a waterfall in a tropical rainforest",
    "a bicycle leaning against a brick wall",
    "a sea turtle swimming over a coral reef",
    "a steaming teapot and two cups on a tray",
    "a red barn in a green meadow with cows",
    "an owl perched on a fence post at night",
    "a plate of sushi on a black slate",
    "a sailboat on the ocean at sunset",
    "a snowy village street with warm lit windows",
    "a close-up of a dew covered spider web",
]
SB_SHOWN = [0, 3, 7, 10, 13, 17, 20, 23, 27, 30, 33, 37]  # the paper's sb columns
NUM_DB = 19


def small(images: torch.Tensor, size: int = 256) -> torch.Tensor:
    """Downsample decoded images for storage."""
    return F.interpolate(images.float(), size=(size, size), mode="area")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def paper_head_mixing(pipe, setup, dev: str, out, num_mixing: int) -> None:
    """The mixing fit with the output head conditioned the way the paper appears to have done it.

    Not the default flow. The default conditions the head on the model's own ``temb`` (captured
    with a hook), which is correct and gives exactly ``(1, 0)`` at the last block. This flow
    rebuilds ``temb`` with the timestep on the 0-1 scale instead of the x1000 the transformer
    applies internally, while the final prediction stays the true model output. It reproduces
    the paper's FLUX coefficients (see "The FLUX head conditioning" in REPORT.md). Same noise,
    prompts and 70/30 split as the default mixing stage. Writes ``mixing_paper_head.json``.
    """
    adapter = DiffusersTransformerAdapter(
        pipe.transformer, block_attrs=("single_transformer_blocks",))
    tr, batch = pipe.transformer, 2
    num_train = int(0.7 * num_mixing)
    train, test = MixingStatistics(adapter.num_layers), MixingStatistics(adapter.num_layers)
    gen = torch.Generator().manual_seed(0)
    t0 = time.time()
    for start in range(0, num_mixing, batch):
        idx = torch.arange(start, min(start + batch, num_mixing))
        adapter.set_conditioning(**setup.conditioning_for(idx))
        z = torch.randn(len(idx), *setup.noise.shape[1:], generator=gen).to(dev, torch.bfloat16)
        task = TransportTask.one_step(len(idx), device=dev, dtype=z.dtype)
        hidden = adapter.hidden_states(z, task)
        # True model output first: calling norm_out below re-triggers the adapter's hook and
        # would replace the captured temb with the rebuilt one.
        final = z.float() - adapter.readout(hidden[-1], task).float()
        pooled = adapter._conditioning_for_batch(len(idx))["pooled_projections"]
        temb = tr.time_text_embed(task.t, pooled)      # timestep 1.0, not 1000
        v = torch.stack([tr.proj_out(tr.norm_out(h, temb)).float() for h in hidden])
        clean = z.float() - v
        is_train = (idx < num_train).to(dev)
        for stats, mask in ((train, is_train), (test, ~is_train)):
            if mask.any():
                stats.update(clean[:, mask], final[mask], z[mask])
        del hidden
    m = evaluate_mixing(train, test)
    save_json({"alpha": m.alpha, "sigma": m.sigma, "held_out_r": m.held_out_r,
               "mean_held_out_r": m.mean_held_out_r, "num_train": m.num_train,
               "num_test": m.num_test}, out / "mixing_paper_head.json")
    log(f"paper-head mixing done in {time.time() - t0:.0f}s, last block "
        f"({m.alpha[-1]:.3f}, {m.sigma[-1]:.3f}), mean R = {m.mean_held_out_r:.4f}")
    adapter.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--num-mixing", type=int, default=256)
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--out-name", default="flux", help="Subfolder of data/ for the results.")
    p.add_argument("--prompts-file", default=None, help="One prompt per line (default: PROMPTS).")
    p.add_argument(
        "--paper-head", action="store_true",
        help="Run only the paper-style head flow (mixing fit with the head conditioned on an "
             "unscaled timestep, see REPORT.md) instead of the default analyses, which use the "
             "model's own conditioning.")
    args = p.parse_args()
    dev, size = args.device, args.size
    out = DATA / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args.model, torch.bfloat16)
    prompts = PROMPTS
    if args.prompts_file:
        prompts = [ln.strip() for ln in open(args.prompts_file) if ln.strip()]
    setup = prepare_text_to_image(pipe, prompts, num_steps=4, height=size, width=size, seed=0,
                                  device=dev, dtype=torch.bfloat16)
    # separate noise (and the same prompts) to fit the ridge probe
    fit_noise = pipe.prepare_latents(8, 16, size, size, torch.bfloat16, dev,
                                     torch.Generator(device=dev).manual_seed(1))[0]
    pipe.transformer.to(dev).eval()
    decode = setup.decode
    log(f"sigmas {setup.sigmas}; noise {tuple(setup.noise.shape)}")
    if args.paper_head:
        paper_head_mixing(pipe, setup, dev, out, args.num_mixing)
        log("paper-head flow done")
        return
    N = setup.num_prompts

    # ------------------------------------------------------------------ all 57 blocks
    adapter = DiffusersTransformerAdapter(pipe.transformer)
    L = adapter.num_layers
    assert L == 57

    if "one_step" not in args.skip:
        # separate noise draws, prompts 0-7, one sample per call to fit in memory
        ridge = fit_ridge_readout(
            adapter, fit_noise, TransportTask.one_step(8, device=dev), batch_size=1,
            on_batch=lambda idx: adapter.set_conditioning(**setup.conditioning_for(idx)),
        )
        rand = make_random_readout(ridge, torch.Generator().manual_seed(0))
        log("ridge probe fitted")
        dist = {k: [] for k in ("model", "ridge", "random", "model_pixel")}
        for i in range(N):
            adapter.set_conditioning(**setup.conditioning_for(torch.tensor([i])))
            z = setup.noise[i : i + 1]
            task = TransportTask.one_step(1, device=dev)
            hidden = adapter.hidden_states(z, task)
            for name, ro in (("model", adapter.readout), ("ridge", ridge), ("random", rand)):
                v = torch.stack([ro(h, task).float() for h in hidden])
                clean = z.float() - v  # t = 1, r = 0
                dist[name].append(layerwise_distances(clean, clean[-1]).cpu())
                if name == "model":
                    pix = torch.cat([decode(clean[j]) for j in range(L)])
                    dist["model_pixel"].append(layerwise_distances(pix[:, None], pix[-1:]).cpu())
                    if i == 0:
                        torch.save({"clean_all_blocks": small(pix)},
                                   out / "one_step_jay_decodes.pt")
            del hidden
            free_cuda()
            log(f"one-step sample {i + 1}/{N}")
        torch.save({k: torch.cat(v, dim=1) for k, v in dist.items()},
                   out / "one_step_distances.pt")

    if "multi_step" not in args.skip:
        probe = LayerwiseProbe(adapter)
        ms = {"times": setup.sigmas, "endpoints": [], "clean": []}
        for i in range(N):
            adapter.set_conditioning(**setup.conditioning_for(torch.tensor([i])))
            steps = probe_sampling_trajectory(probe, setup.noise[i : i + 1], setup.sigmas)
            final = steps[-1].next_state
            ms["endpoints"].append(
                torch.stack([layerwise_distances(r.endpoints, final).cpu() for r in steps]))
            ms["clean"].append(
                torch.stack([layerwise_distances(r.clean, final).cpu() for r in steps]))
            del steps
            free_cuda()
            log(f"4-step sample {i + 1}/{N}")
        ms["endpoints"] = torch.cat(ms["endpoints"], dim=2)  # (steps, L, N)
        ms["clean"] = torch.cat(ms["clean"], dim=2)
        torch.save(ms, out / "multi_step_distances.pt")

    if "two_step" not in args.skip:
        probe = LayerwiseProbe(adapter)
        adapter.set_conditioning(**setup.conditioning_for(torch.tensor([0])))
        steps = probe_sampling_trajectory(probe, setup.noise[:1], [1.0, 0.5, 0.0])
        sb = [NUM_DB + k for k in range(38)]
        torch.save(
            {"endpoints": [small(decode(r.endpoints[sb, 0])) for r in steps],
             "clean": [small(decode(r.clean[sb, 0])) for r in steps]},
            out / "two_step_jay_decodes.pt",
        )
        del steps
        free_cuda()
        log("two-step decodes done")
    adapter.close()

    # ------------------------------------------------------------- 38 single-stream blocks
    if "mixing" not in args.skip:
        adapter = DiffusersTransformerAdapter(
            pipe.transformer, block_attrs=("single_transformer_blocks",))
        probe = LayerwiseProbe(adapter)
        t0 = time.time()
        mixing = run_mixing_analysis(
            probe, tuple(setup.noise.shape[1:]), args.num_mixing, batch_size=2, seed=0,
            device=dev, dtype=torch.bfloat16,
            on_batch=lambda idx: adapter.set_conditioning(**setup.conditioning_for(idx)),
        )
        save_json({"alpha": mixing.alpha, "sigma": mixing.sigma,
                   "held_out_r": mixing.held_out_r, "mean_held_out_r": mixing.mean_held_out_r,
                   "num_train": mixing.num_train, "num_test": mixing.num_test},
                  out / "mixing.json")
        log(f"mixing done in {time.time() - t0:.0f}s, mean R = {mixing.mean_held_out_r:.4f}")

        adapter.set_conditioning(**setup.conditioning_for(torch.tensor([0])))
        z = setup.noise[:1]
        clean = probe.run(z, TransportTask.one_step(1, device=dev)).clean[:, 0]
        a = mixing.alpha.to(dev).float().view(-1, 1, 1)
        s = mixing.sigma.to(dev).float().view(-1, 1, 1)
        fitted = a * clean[-1] + s * z[0].float()
        torch.save({"actual": small(decode(clean[SB_SHOWN])),
                    "fitted": small(decode(fitted[SB_SHOWN])),
                    "layers": SB_SHOWN}, out / "mixing_example.pt")
        adapter.close()

    log("all FLUX analyses done")


if __name__ == "__main__":
    with torch.no_grad():
        main()
