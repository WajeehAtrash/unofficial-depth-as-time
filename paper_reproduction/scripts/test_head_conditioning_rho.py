"""Cross-check of the head-conditioning finding on the paper's rho figures (Fig. 3, Fig. 7).

Recomputes one-step and four-step rho(i) for FLUX with the output head conditioned on
(a) the temb the model used and (b) a temb rebuilt with the timestep on the 0-1 scale
(not multiplied by 1000), and compares both with the curves digitized from the paper.
The sampler itself always uses the true model output.
"""

from __future__ import annotations

import json

import numpy as np
import torch

import common
from depth_as_time import TransportTask
from depth_as_time.metrics import layerwise_distances, normalized_remaining_from_distances
from depth_as_time.models.diffusers_adapter import DiffusersTransformerAdapter
from depth_as_time.text_to_image import load_pipeline, prepare_text_to_image
from run_flux import PROMPTS, log

NUM_DB = 19
RASTER = json.loads((common.ROOT / "paper_data/paper_raster_data.json").read_text())


def mae_pp(ours, paper) -> float:
    return float(np.abs(np.asarray(ours) - np.asarray(paper)).mean() * 100)


@torch.no_grad()
def main() -> None:
    dev, size = "cuda:0", 1024
    pipe = load_pipeline("black-forest-labs/FLUX.1-schnell", torch.bfloat16)
    setup = prepare_text_to_image(pipe, PROMPTS, num_steps=4, height=size, width=size, seed=0,
                                  device=dev, dtype=torch.bfloat16)
    tr = pipe.transformer.to(dev).eval()
    adapter = DiffusersTransformerAdapter(tr)
    sig = setup.sigmas
    names = ("true", "unscaled")
    one = {k: [] for k in names}
    four_r = {k: [] for k in names}
    four_0 = {k: [] for k in names}
    for i in range(setup.num_prompts):
        adapter.set_conditioning(**setup.conditioning_for(torch.tensor([i])))
        pooled = adapter._conditioning_for_batch(1)["pooled_projections"]
        z = setup.noise[i : i + 1]
        ends = {k: [] for k in names}
        cleans = {k: [] for k in names}
        for k in range(4):
            t, r = sig[k], sig[k + 1]
            task = TransportTask.constant(t, r, 1, device=dev)
            hidden = adapter.hidden_states(z, task)
            tembs = {
                "true": adapter._capture.temb,
                "unscaled": tr.time_text_embed(
                    torch.full((1,), t, device=dev, dtype=torch.bfloat16), pooled),
            }
            v_true_last = None
            for name in names:
                v = torch.stack([tr.proj_out(tr.norm_out(h, tembs[name])).float() for h in hidden])
                ends[name].append((z.float() - (t - r) * v).cpu())
                cleans[name].append((z.float() - t * v).cpu())
                if name == "true":
                    v_true_last = v[-1]
            del hidden
            z = (z.float() - (t - r) * v_true_last).to(torch.bfloat16)  # true sampler update
        final = z.float().cpu()
        for name in names:
            c0 = cleans[name][0]
            one[name].append(layerwise_distances(c0, c0[-1]))
            four_r[name].append(torch.stack([layerwise_distances(e, final) for e in ends[name]]))
            four_0[name].append(torch.stack([layerwise_distances(c, final) for c in cleans[name]]))
        common.free_cuda()
        log(f"sample {i + 1}/{setup.num_prompts}")
    out = {}
    p3 = RASTER["fig3_rho"]["red"]
    pb = RASTER["figB1_controls"]
    for name in names:
        d1 = torch.cat(one[name], dim=1)
        rho1 = normalized_remaining_from_distances([d1])[0].numpy()
        res = {"one_step_rho": rho1.tolist(), "fig3_mae_pp": mae_pp(rho1, p3),
               "figB1_blue_mae_pp": mae_pp(rho1, pb["blue"]),
               "figB1_red_mae_pp": mae_pp(rho1, pb["red"])}
        for key, dist, pkey in (("x_r", four_r, "fig7_flux_xr"), ("x_0", four_0, "fig7_flux_x0")):
            d = torch.cat(dist[name], dim=2)[:, NUM_DB:]           # (steps, 38, N)
            rho = torch.stack(normalized_remaining_from_distances(list(d))).numpy()
            paper = np.array(RASTER[pkey]["steps_sb38"])
            res[f"fig7_{key}_rho"] = rho.tolist()
            res[f"fig7_{key}_mae_pp"] = mae_pp(rho, paper)
            res[f"fig7_{key}_per_step_pp"] = [mae_pp(rho[s], paper[s]) for s in range(4)]
        out[name] = res
        log(f"{name:9s} Fig3 {res['fig3_mae_pp']:.2f} pp | B.1 blue {res['figB1_blue_mae_pp']:.2f}"
            f" red {res['figB1_red_mae_pp']:.2f} | Fig7 x_r {res['fig7_x_r_mae_pp']:.2f} "
            f"{np.round(res['fig7_x_r_per_step_pp'], 1).tolist()} | Fig7 x_0 "
            f"{res['fig7_x_0_mae_pp']:.2f} {np.round(res['fig7_x_0_per_step_pp'], 1).tolist()}")
    common.save_json(out, common.DATA / "flux_head_conditioning_rho.json")


if __name__ == "__main__":
    main()
