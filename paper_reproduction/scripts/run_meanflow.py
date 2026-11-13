"""Run every MeanFlow (ImageNet 256, SiT) analysis of the paper and save raw results.

Outputs go to ``data/meanflow/``. Figures are made separately by ``make_figures.py``.
"""

from __future__ import annotations

import argparse
import time

import torch

from common import DATA, MEANFLOW_REPO, save_json
from depth_as_time import LayerwiseProbe, TransportTask, probe_sampling_trajectory
from depth_as_time.metrics import layerwise_distances
from depth_as_time.mixing import run_mixing_analysis
from depth_as_time.readouts import fit_ridge_readout, make_random_readout
from meanflow_adapter import MeanFlowAdapter, RMSOnlyReadout, load_meanflow, make_vae_decoder

# Classes for the visual rows, chosen to match the paper's examples.
PANDA, JAY, MONARCH, SPANIEL = 388, 17, 323, 156


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def draw(n: int, seed: int, device: str, labels=None):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 4, 32, 32, generator=g)
    y = torch.randint(0, 1000, (n,), generator=g) if labels is None else torch.as_tensor(labels)
    return z.to(device), y.to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(MEANFLOW_REPO / "meanflow-B4.pth"))
    p.add_argument("--model", default="SiT-B/4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-rho", type=int, default=32)
    p.add_argument("--num-mixing", type=int, default=256)
    p.add_argument("--num-centering", type=int, default=192)
    args = p.parse_args()
    out = DATA / "meanflow"
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device

    sit = load_meanflow(args.ckpt, args.model, dev)
    model = MeanFlowAdapter(sit).eval()
    decode = make_vae_decoder(dev)
    probe = LayerwiseProbe(model)
    L = model.num_layers

    # ---- sanity: the last-layer readout equals the original network output
    z, y = draw(4, 123, dev)
    task = TransportTask.one_step(4, device=dev)
    with torch.no_grad():
        ref = sit(z, task.t, task.interval, y)
        ours = model(z, task, y)
    err = (ref - ours).abs().max().item()
    log(f"adapter check: max |sit - adapter| = {err:.2e}")
    assert err < 1e-4

    # ---- 1) one-step rho (Fig. 3, App. B.1 controls), latent and pixel space
    z, y = draw(args.num_rho, 0, dev)
    task = TransportTask.one_step(z.shape[0], device=dev)
    zf, yf = draw(32, 1, dev)  # separate fitting set for the ridge probe
    ridge = fit_ridge_readout(model, zf, TransportTask.one_step(32, device=dev), yf)
    rand = make_random_readout(ridge, torch.Generator().manual_seed(0))
    one_step = {}
    for name, readout in (("model", None), ("ridge", ridge), ("random", rand)):
        res = LayerwiseProbe(model, readout).run(z, task, y)
        clean = res.clean
        entry = {"latent": layerwise_distances(clean, clean[-1]).cpu()}
        pix = decode(clean.flatten(0, 1)).reshape(L, z.shape[0], 3, 256, 256)
        entry["pixel"] = layerwise_distances(pix, pix[-1]).cpu()
        one_step[name] = entry
        log(f"one-step rho [{name}] done")
    torch.save(one_step, out / "one_step_distances.pt")

    # visual: one-step clean decodes, panda (Fig. 3 middle row, Fig. 8b top row)
    zv, yv = draw(1, 7, dev, [PANDA])
    res = probe.run(zv, TransportTask.one_step(1, device=dev), yv)
    torch.save({"clean": decode(res.clean[:, 0])}, out / "one_step_panda_decodes.pt")

    # ---- 2) mixing schedule, 256 trajectories, 179/77 split (Fig. 4, 5, B.2)
    t0 = time.time()
    mixing = run_mixing_analysis(
        probe, (4, 32, 32), args.num_mixing, batch_size=64, num_classes=1000, seed=0, device=dev
    )
    save_json(
        {
            "alpha": mixing.alpha, "sigma": mixing.sigma, "held_out_r": mixing.held_out_r,
            "mean_held_out_r": mixing.mean_held_out_r,
            "num_train": mixing.num_train, "num_test": mixing.num_test,
        },
        out / "mixing.json",
    )
    log(f"mixing done in {time.time() - t0:.0f}s: mean R = {mixing.mean_held_out_r:.4f}")

    # centering ablation (App. B.2), N = 192, same protocol
    abl = {}
    for name, readout in (("layernorm", None), ("rms_only", RMSOnlyReadout(model))):
        m = run_mixing_analysis(
            LayerwiseProbe(model, readout), (4, 32, 32), args.num_centering, batch_size=64,
            num_classes=1000, seed=0, device=dev,
        )
        abl[name] = {"alpha": m.alpha, "sigma": m.sigma, "held_out_r": m.held_out_r,
                     "mean_held_out_r": m.mean_held_out_r}
        log(f"centering [{name}] mean R = {m.mean_held_out_r:.4f}")
    save_json(abl, out / "centering_ablation.json")

    # reconstruction example (Fig. 4): actual x_i, fitted alpha_i x_L + sigma_i eps, residual
    zv, yv = draw(1, 11, dev, [SPANIEL])
    clean = probe.run(zv, TransportTask.one_step(1, device=dev), yv).clean[:, 0]
    a = mixing.alpha.to(dev).view(-1, 1, 1, 1).float()
    s = mixing.sigma.to(dev).view(-1, 1, 1, 1).float()
    fitted = a * clean[-1] + s * zv[0]
    torch.save(
        {"actual_latent": clean.cpu(), "fitted_latent": fitted.cpu(),
         "actual": decode(clean), "fitted": decode(fitted)},
        out / "mixing_example.pt",
    )

    # ---- 3) three-step MeanFlow (Fig. 8b): x_r and x_0 rho + decodes
    times = [1.0, 2 / 3, 1 / 3, 0.0]
    z, y = draw(args.num_rho, 0, dev)
    steps = probe_sampling_trajectory(probe, z, times, y)
    final = steps[-1].next_state
    ms = {"times": times}
    for kind in ("endpoints", "clean"):
        ms[kind] = [layerwise_distances(getattr(r, kind), final).cpu() for r in steps]
    torch.save(ms, out / "multi_step_distances.pt")
    vis = {}
    for name, cls, seed in (("jay", JAY, 3), ("monarch", MONARCH, 5)):
        zv, yv = draw(1, seed, dev, [cls])
        st = probe_sampling_trajectory(probe, zv, times, yv)
        vis[name] = {
            "endpoints": [decode(r.endpoints[:, 0]) for r in st],
            "clean": [decode(r.clean[:, 0]) for r in st],
        }
    torch.save(vis, out / "multi_step_decodes.pt")
    log("all MeanFlow analyses done")


if __name__ == "__main__":
    main()
