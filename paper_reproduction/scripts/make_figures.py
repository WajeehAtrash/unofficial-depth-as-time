"""Regenerate the paper's FLUX-schnell and MeanFlow figures from ``data/`` and compare with the
digitized paper values in ``paper_data/``.

Writes ``figures/*.png``, ``comparison.json``, ``sensitivity.json`` (FLUX variant runs) and
``summary_table.md``.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import DATA, FIGURES, ROOT, save_json  # noqa: E402
from depth_as_time.metrics import normalized_remaining_from_distances  # noqa: E402

RED, TEAL, BLUE, GREY = "#B02418", "#004E64", "#3A78B5", "#8A8A8A"
PAPER_STYLE = dict(color="black", lw=1.2, ls="--", alpha=0.75)
SB_SHOWN = [0, 3, 7, 10, 13, 17, 20, 23, 27, 30, 33, 37]
NUM_DB = 19

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

paper_vec = json.loads((ROOT / "paper_data/paper_vector_data.json").read_text())
paper_ras = json.loads((ROOT / "paper_data/paper_raster_data.json").read_text())
cmp: dict = {}


# ----------------------------------------------------------------------------- helpers
def rho(dist: torch.Tensor) -> np.ndarray:
    """One-step rho(i) from stored distances ``(L, N)`` to the last layer."""
    return normalized_remaining_from_distances([dist.float()])[0].numpy()


def rho_multi(dist: torch.Tensor) -> np.ndarray:
    """Multi-step rho(i) from distances ``(steps, L, N)`` to the final sample (shared max)."""
    return torch.stack(normalized_remaining_from_distances(list(dist.float()))).numpy()


def rho_band(dist: torch.Tensor):
    """rho(i) and its standard error over trajectories, for shaded bands."""
    d = dist.float()
    r = d / d.amax(dim=0, keepdim=True).clamp_min(1e-30)
    return r.mean(1).numpy(), (r.std(1) / np.sqrt(r.shape[1])).numpy()


def curve_stats(ours, paper) -> dict:
    ours, paper = np.asarray(ours, float), np.asarray(paper, float)
    diff = ours - paper
    return {
        "mae_pct_points": float(np.mean(np.abs(diff)) * 100),
        "max_abs_pct_points": float(np.max(np.abs(diff)) * 100),
        "relative_mae_pct": float(np.mean(np.abs(diff)) / np.mean(np.abs(paper)) * 100),
        "pearson_r": float(np.corrcoef(ours, paper)[0, 1]),
    }


def coeff_stats(ours, paper) -> dict:
    ours, paper = np.asarray(ours, float), np.asarray(paper, float)
    keep = np.abs(paper) > 0.05  # relative error is meaningless near 0
    rel = np.abs(ours - paper)[keep] / np.abs(paper)[keep] * 100
    return {
        "mean_abs_diff": float(np.mean(np.abs(ours - paper))),
        "mean_relative_pct": float(rel.mean()),
        "max_relative_pct": float(rel.max()),
        "pearson_r": float(np.corrcoef(ours, paper)[0, 1]),
    }


def rel_pct(ours: float, paper: float) -> float:
    return float((ours - paper) / abs(paper) * 100)


def to_img(t: torch.Tensor) -> np.ndarray:
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


def image_row(axs, images, labels, label_color=None):
    for ax, img, lab in zip(axs, images, labels):
        ax.imshow(to_img(img))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(lab, fontsize=8, color=label_color or "dimgray", pad=2)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


def exists(*parts) -> bool:
    return (DATA.joinpath(*parts)).exists()


# ---------------------------------------------------------------- MeanFlow numbers
mf_mix = json.loads((DATA / "meanflow/mixing.json").read_text())
mf_cent = json.loads((DATA / "meanflow/centering_ablation.json").read_text())
mf_one = torch.load(DATA / "meanflow/one_step_distances.pt")
p_mf = paper_vec["coeffs_meanflow"]
p_R = paper_vec["heldout_R_meanflow_layernorm"]["R"]
p_R_rms = paper_vec["heldout_R_meanflow_rmsonly"]["R"][:11]
ours_R = mf_mix["held_out_r"][:11]
# The paper's Fig. B.3 R curve comes from the N=192 centering-ablation run: compare like for like
ours_R192 = mf_cent["layernorm"]["held_out_r"][:11]
cmp["meanflow"] = {
    "alpha_block4": {"paper_text": 2.18, "ours": mf_mix["alpha"][4],
                     "diff_pct": rel_pct(mf_mix["alpha"][4], 2.18)},
    "sigma_block4": {"paper_text": -6.04, "ours": mf_mix["sigma"][4],
                     "diff_pct": rel_pct(mf_mix["sigma"][4], -6.04)},
    "mean_heldout_R": {"paper_text": 0.10, "paper_fig_N192": float(np.mean(p_R)),
                       "ours_N192": float(np.mean(ours_R192)),
                       "diff_pct_vs_fig": rel_pct(float(np.mean(ours_R192)), float(np.mean(p_R))),
                       "ours_N256": mf_mix["mean_held_out_r"],
                       "diff_pct_N256_vs_fig": rel_pct(mf_mix["mean_held_out_r"],
                                                       float(np.mean(p_R)))},
    "alpha_curve": coeff_stats(mf_mix["alpha"], p_mf["alpha"]),
    "sigma_curve": coeff_stats(mf_mix["sigma"], p_mf["sigma"]),
    "heldout_R_curve": coeff_stats(ours_R192, p_R),
    "heldout_R_curve_N256": coeff_stats(ours_R, p_R),
    "centering_ablation": {
        "ours_mean_R_layernorm_N192": mf_cent["layernorm"]["mean_held_out_r"],
        "ours_mean_R_rms_only_N192": mf_cent["rms_only"]["mean_held_out_r"],
        "paper_mean_R_layernorm": float(np.mean(p_R)),
        "paper_mean_R_rms_only": float(np.mean(p_R_rms)),
        "ours_rms_vs_layernorm_change_pct": rel_pct(mf_cent["rms_only"]["mean_held_out_r"],
                                                    mf_cent["layernorm"]["mean_held_out_r"]),
        "paper_rms_vs_layernorm_change_pct": rel_pct(float(np.mean(p_R_rms)),
                                                     float(np.mean(p_R))),
        "rms_only_R_curve": coeff_stats(mf_cent["rms_only"]["held_out_r"][:11], p_R_rms),
    },
}

# ---------------------------------------------------------- Fig. 5: MeanFlow coefficients
fig, ax = plt.subplots(figsize=(5.2, 3.4))
blocks = np.arange(12)
ax.plot(blocks, mf_mix["alpha"], "-o", color=TEAL, label=r"ours $\alpha_i$")
ax.plot(blocks, mf_mix["sigma"], "-o", color=RED, label=r"ours $\sigma_i$")
ax.plot(blocks, p_mf["alpha"], marker="x", **PAPER_STYLE, label="paper (digitized)")
ax.plot(blocks, p_mf["sigma"], marker="x", **PAPER_STYLE)
ax.set_xlabel("network depth (block index)")
ax.set_ylabel("coefficient value")
ax.set_title("MeanFlow recovered depthwise mixing schedule (Fig. 5)")
ax.legend(frameon=False, fontsize=8)
save(fig, "fig5_meanflow_mixing_coefficients.png")

# ------------------------------------------------------- Fig. B.3: centering ablation
fig, ax = plt.subplots(figsize=(5, 3.4))
b1 = np.arange(1, 12)
ax.plot(b1, mf_cent["layernorm"]["held_out_r"][:11], "-o", color=BLUE,
        label="ours, LayerNorm (N=192)")
ax.plot(b1, mf_cent["rms_only"]["held_out_r"][:11], "--x", color="C1",
        label="ours, RMS-only (N=192)")
ax.plot(b1, ours_R, "-", color=TEAL, lw=1, label="ours, LayerNorm (N=256)")
ax.plot(b1, p_R, marker="s", ms=3, **PAPER_STYLE, label="paper, LayerNorm")
ax.set_xlabel("block")
ax.set_ylabel("held-out R (2-coef fit)")
ax.set_ylim(0, 0.2)
ax.set_title("MeanFlow centering ablation (Fig. B.3)")
ax.legend(frameon=False, fontsize=8)
save(fig, "figB3_meanflow_centering_ablation.png")

# ------------------------------------------------- MeanFlow readout controls (extra)
fig, ax = plt.subplots(figsize=(5, 3.4))
x = np.arange(12) / 11
for name, col, ls in (("model", RED, "-"), ("ridge", BLUE, "--"), ("random", GREY, ":")):
    m, se = rho_band(mf_one[name]["latent"])
    ax.plot(x, m, ls, color=col, label={"model": r"$W_{out}$", "ridge": "trained probe",
                                        "random": "random probe"}[name])
    ax.fill_between(x, m - se, m + se, color=col, alpha=0.15)
ax.set_xlabel("normalized depth i/L")
ax.set_ylabel(r"normalized denoising remaining $\rho(i)$")
ax.set_title("MeanFlow readout controls (not in paper, ours only)")
ax.legend(frameon=False, fontsize=8)
save(fig, "extra_meanflow_readout_controls.png")
cmp["meanflow"]["one_step_rho_latent"] = rho(mf_one["model"]["latent"]).tolist()
cmp["meanflow"]["one_step_rho_pixel"] = rho(mf_one["model"]["pixel"]).tolist()

# ---------------------------------------------------- Fig. 8b: MeanFlow ImageNet decodes
panda = torch.load(DATA / "meanflow/one_step_panda_decodes.pt")["clean"]
ms_vis = torch.load(DATA / "meanflow/multi_step_decodes.pt")
fig = plt.figure(figsize=(13, 9.3))
gs = fig.add_gridspec(7, 12, hspace=0.25, wspace=0.03)
image_row([fig.add_subplot(gs[0, j]) for j in range(12)], panda, [str(j) for j in range(12)])
fig.text(0.1, 0.86, "one-step\n" + r"$\tilde x_0^i$", ha="right", va="center", fontsize=9)
for k in range(3):
    image_row([fig.add_subplot(gs[1 + k, j]) for j in range(12)],
              ms_vis["jay"]["endpoints"][k], [""] * 12)
    fig.text(0.1, 0.745 - 0.113 * k, f"step {k + 1}\n" + r"$\tilde x_r^i$", ha="right",
             va="center", fontsize=9)
for k in range(3):
    image_row([fig.add_subplot(gs[4 + k, j]) for j in range(12)],
              ms_vis["monarch"]["clean"][k], [""] * 12)
    fig.text(0.1, 0.405 - 0.113 * k, f"step {k + 1}\n" + r"$\tilde x_0^i$", ha="right",
             va="center", fontsize=9)
fig.suptitle("MeanFlow (SiT-B/4, ImageNet 256): one-step decodes, and 3-step (1, 2/3, 1/3, 0) "
             r"$\tilde x_r$ / $\tilde x_0$ decodes (Fig. 8b)", y=0.93)
save(fig, "fig8b_meanflow_imagenet_decodes.png")

# MeanFlow 3-step rho (no counterpart curve in the paper for ImageNet; CIFAR-10 only)
mf_ms = torch.load(DATA / "meanflow/multi_step_distances.pt")
fig, ax = plt.subplots(figsize=(6, 3.4))
for key, col, lab in (("endpoints", RED, r"$\tilde x_r^i$"), ("clean", TEAL, r"$\tilde x_0^i$")):
    curves = rho_multi(torch.stack(mf_ms[key]))
    for s in range(3):
        xs = (s + np.arange(12) / 12) / 3
        ax.plot(xs, curves[s], color=col, label=lab if s == 0 else None)
for s in (1, 2):
    ax.axvline(s / 3, color=RED, ls="--", lw=0.6, alpha=0.6)
ax.set_xlabel("normalized progress (step + i/L)")
ax.set_ylabel(r"$\rho(i)$ to final sample")
ax.set_title("MeanFlow 3-step: denoise-then-renoise in $\\tilde x_r$ (ours)")
ax.legend(frameon=False)
save(fig, "fig8_meanflow_three_step_rho.png")
xr = rho_multi(torch.stack(mf_ms["endpoints"]))
cmp["meanflow"]["three_step_xr"] = {
    f"step{s + 1}": {"start": float(xr[s, 0]), "min": float(xr[s].min()),
                     "argmin_block": int(xr[s].argmin()), "end": float(xr[s, -1])}
    for s in range(3)
}

# --------------------------------------------------------------------- FLUX numbers
have_flux_one = exists("flux", "one_step_distances.pt")
have_flux_ms = exists("flux", "multi_step_distances.pt")
have_flux_mix = exists("flux", "mixing.json")
cmp["flux"] = {}

if have_flux_mix:
    fx_mix = json.loads((DATA / "flux/mixing.json").read_text())
    p_fx = paper_vec["coeffs_flux_sb"]
    cmp["flux"]["alpha_curve"] = coeff_stats(fx_mix["alpha"], p_fx["alpha"])
    cmp["flux"]["sigma_curve"] = coeff_stats(fx_mix["sigma"], p_fx["sigma"])
    cmp["flux"]["alpha_curve_excl_last"] = coeff_stats(fx_mix["alpha"][:-1], p_fx["alpha"][:-1])
    cmp["flux"]["sigma_curve_excl_last"] = coeff_stats(fx_mix["sigma"][:-1], p_fx["sigma"][:-1])
    cmp["flux"]["mean_heldout_R"] = fx_mix["mean_held_out_r"]
    cmp["flux"]["sigma_block0"] = {"paper_fig": p_fx["sigma"][0], "ours": fx_mix["sigma"][0],
                                   "diff_pct": rel_pct(fx_mix["sigma"][0], p_fx["sigma"][0])}

    # ---- Fig. B.2: all coefficients (MeanFlow + FLUX panels)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
    for ax, ours, paper, title in ((axs[0], mf_mix, p_mf, "MeanFlow"),
                                   (axs[1], fx_mix, p_fx, "Distilled (FLUX, sb blocks)")):
        b = np.arange(len(ours["alpha"]))
        ax.plot(b, ours["alpha"], "-o", ms=3, color=TEAL, label=r"ours $\alpha$")
        ax.plot(b, ours["sigma"], "-o", ms=3, color=RED, label=r"ours $\sigma$")
        ax.plot(b, paper["alpha"], **PAPER_STYLE, label="paper (digitized)")
        ax.plot(b, paper["sigma"], **PAPER_STYLE)
        ax.set_title(title)
        ax.set_xlabel("block")
    axs[0].set_ylabel("coefficient value")
    axs[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Recovered depthwise mixing coefficients (Fig. B.2)", y=1.02)
    save(fig, "figB2_mixing_coefficients_meanflow_flux.png")

    # ---- Fig. 4: mixing reconstructions (MeanFlow and FLUX)
    mf_ex = torch.load(DATA / "meanflow/mixing_example.pt")
    fx_ex = torch.load(DATA / "flux/mixing_example.pt")
    fig = plt.figure(figsize=(13, 8.4))
    gs = fig.add_gridspec(7, 12, hspace=0.12, wspace=0.03,
                          height_ratios=[1, 1, 1, 0.45, 1, 1, 1])
    for row0, ex, labels, name in (
        (0, mf_ex, [str(j) for j in range(12)], "MeanFlow (transport task)"),
        (4, fx_ex, [f"sb{j:02d}" for j in SB_SHOWN], "FLUX.1-schnell (distilled)"),
    ):
        actual, fitted = ex["actual"], ex["fitted"]
        resid = (actual - fitted).abs().mean(dim=1)
        vmax = float(resid.quantile(0.99))
        top = [fig.add_subplot(gs[row0, j]) for j in range(12)]
        image_row(top, actual, labels)
        mid = [fig.add_subplot(gs[row0 + 1, j]) for j in range(12)]
        image_row(mid, fitted, [""] * 12)
        bot = []
        for j in range(12):
            ax = fig.add_subplot(gs[row0 + 2, j])
            ax.imshow(resid[j].numpy(), cmap="magma", vmin=0, vmax=vmax)
            ax.axis("off")
            bot.append(ax)
        for axes, lab in ((top, "actual $x_i$"), (mid, r"fitted $\hat x_i$"),
                          (bot, r"$|x_i-\hat x_i|$")):
            pos = axes[0].get_position()
            fig.text(pos.x0 - 0.01, (pos.y0 + pos.y1) / 2, lab, ha="right", va="center",
                     fontsize=9)
        pos = top[0].get_position()
        fig.text(0.5, pos.y1 + 0.03, name, ha="center", color=RED, fontsize=11)
    fig.suptitle("Depthwise mixing reconstructions on a held-out trajectory (Fig. 4)", y=0.97)
    save(fig, "fig4_mixing_reconstructions.png")

    # held-out R per block, MeanFlow and FLUX (extra panel; paper reports only MeanFlow/Drifting R)
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.plot(np.arange(1, 12) / 12, ours_R, "-o", ms=3, color=RED, label="MeanFlow (ours)")
    ax.plot(np.arange(1, 12) / 12, p_R, marker="s", ms=3, **PAPER_STYLE,
            label="MeanFlow (paper)")
    fr = fx_mix["held_out_r"][:-1]
    ax.plot(np.arange(1, 38) / 38, fr, "-o", ms=2, color=TEAL, label="FLUX sb (ours)")
    ax.set_xlabel("normalized depth i/L")
    ax.set_ylabel("held-out R")
    ax.set_title("Held-out mixing reconstruction error")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "extra_heldout_R_meanflow_flux.png")

if have_flux_one:
    fx_one = torch.load(DATA / "flux/one_step_distances.pt")
    L = 57
    x = np.arange(L) / (L - 1)
    p3 = np.array(paper_ras["fig3_rho"]["red"])
    pB = paper_ras["figB1_controls"]
    ours_model = rho(fx_one["model"])
    cmp["flux"]["one_step_rho"] = {
        "W_out_vs_paper_fig3_red": curve_stats(ours_model, p3),
        "W_out_vs_paper_figB1_Wout_blue": curve_stats(ours_model, pB["blue"]),
        "ridge_vs_paper_figB1_trained_red": curve_stats(rho(fx_one["ridge"]), pB["red"]),
        "SWAPPED_W_out_vs_paper_figB1_red": curve_stats(ours_model, pB["red"]),
        "SWAPPED_ridge_vs_paper_figB1_blue": curve_stats(rho(fx_one["ridge"]), pB["blue"]),
        "random_vs_paper_figB1_random_grey": curve_stats(rho(fx_one["random"]), pB["grey"]),
        "W_out_pixel_space_vs_paper_figB1_Wout": curve_stats(rho(fx_one["model_pixel"]),
                                                             pB["blue"]),
        "ours_W_out_vs_ours_ridge": curve_stats(ours_model, rho(fx_one["ridge"])),
        "paper_W_out_vs_paper_trained": curve_stats(pB["blue"], pB["red"]),
    }

    # ---- Fig. 3: one-step decodes (FLUX jay, MeanFlow panda) + rho curve
    jay = torch.load(DATA / "flux/one_step_jay_decodes.pt")["clean_all_blocks"]
    fig = plt.figure(figsize=(16, 3.6))
    gs = fig.add_gridspec(2, 14, width_ratios=[1] * 12 + [0.9, 4.2], wspace=0.03, hspace=0.3)
    image_row([fig.add_subplot(gs[0, j]) for j in range(12)],
              jay[[NUM_DB + k for k in SB_SHOWN]], [f"sb{k:02d}" for k in SB_SHOWN])
    image_row([fig.add_subplot(gs[1, j]) for j in range(12)], panda, [str(j) for j in range(12)])
    fig.text(0.115, 0.68, "FLUX.1-schnell\n(distilled)", ha="right", color=RED, fontsize=9)
    fig.text(0.115, 0.25, "MeanFlow\n(transport task)", ha="right", color=RED, fontsize=9)
    ax = fig.add_subplot(gs[:, 13])
    m, se = rho_band(fx_one["model"])
    ax.plot(x, m, color=RED, label=r"FLUX-schnell $W_{out}$ (ours)")
    ax.fill_between(x, m - se, m + se, color=RED, alpha=0.15)
    ax.plot(x, p3, **PAPER_STYLE, label="paper Fig. 3 (digitized)")
    ax.axvline(NUM_DB / L, color=RED, ls="--", lw=0.6)
    ax.text(0.03, 0.05, "db", fontsize=8)
    ax.text(NUM_DB / L + 0.03, 0.05, "sb", fontsize=8)
    ax.set_xlabel("normalized progress i/L")
    ax.set_ylabel(r"$\rho(i)$")
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Depthwise denoising-like progression in one-step generators (Fig. 3)", y=1.02)
    save(fig, "fig3_one_step_depth_as_time.png")

    # ---- Fig. B.1: readout controls on FLUX
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for key, col, ls, lab, pkey in (("model", BLUE, "--", r"$W_{out}$", "blue"),
                                    ("ridge", RED, "-", "trained probe", "red"),
                                    ("random", GREY, ":", "random probe", "grey")):
        m, se = rho_band(fx_one[key])
        ax.plot(x, m, ls, color=col, lw=1.8, label=f"{lab} (ours)")
        ax.fill_between(x, m - se, m + se, color=col, alpha=0.15)
        ax.plot(x, pB[pkey], color=col, lw=0.9, alpha=0.6, marker=".", ms=2,
                label=f"{lab} (paper)")
    ax.axvline(NUM_DB / L, color=RED, ls="--", lw=0.6)
    ax.set_xlabel("normalized progress i/L")
    ax.set_ylabel(r"$\rho(i)$")
    ax.set_title("Readout controls on one-step FLUX-schnell (Fig. B.1)\n"
                 "paper curves drawn with the paper's legend labels", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "figB1_flux_readout_controls.png")

    # latent vs pixel space (extra)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(x, ours_model, color=RED, label="ours, latent-space L2")
    ax.plot(x, rho(fx_one["model_pixel"]), color=TEAL, label="ours, pixel-space L2")
    ax.plot(x, pB["blue"], **PAPER_STYLE, label=r"paper $W_{out}$ (Fig. B.1)")
    ax.set_xlabel("normalized progress i/L")
    ax.set_ylabel(r"$\rho(i)$")
    ax.set_title("FLUX one-step rho: distance space")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "extra_flux_rho_latent_vs_pixel.png")

if have_flux_ms:
    fx_ms = torch.load(DATA / "flux/multi_step_distances.pt")
    # The paper's Fig. 7 matches the 38 single-stream blocks (as in Fig. 6), not all 57 blocks.
    for key, pkey in (("endpoints", "fig7_flux_xr"), ("clean", "fig7_flux_x0")):
        ours57 = rho_multi(fx_ms[key])
        paper57 = np.array(paper_ras[pkey]["steps"])
        cmp["flux"][f"four_step_{key}_all57_blocks"] = {
            "all_steps": curve_stats(ours57.ravel(), paper57.ravel()),
            **{f"step{s + 1}": curve_stats(ours57[s], paper57[s]) for s in range(4)},
        }
    L = 38
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, key, pkey, lab in ((axs[0], "endpoints", "fig7_flux_xr", r"$\tilde x_r^i$"),
                               (axs[1], "clean", "fig7_flux_x0", r"$\tilde x_0^i$")):
        ours = rho_multi(fx_ms[key][:, NUM_DB:])
        paper = np.array(paper_ras[pkey]["steps_sb38"])
        cmp["flux"][f"four_step_{key}"] = {
            "all_steps": curve_stats(ours.ravel(), paper.ravel()),
            **{f"step{s + 1}": curve_stats(ours[s], paper[s]) for s in range(4)},
        }
        for s in range(4):
            xs = (s + np.arange(L) / L) / 4
            ax.plot(xs, ours[s], color=RED, label="ours" if s == 0 else None)
            ax.plot(xs, paper[s], **PAPER_STYLE, label="paper (digitized)" if s == 0 else None)
            if s:
                ax.axvline(s / 4, color=RED, ls="--", lw=0.6, alpha=0.6)
        ax.set_title(f"FLUX 4-step, {lab} prediction")
        ax.set_xlabel("normalized progress (step + i/L)")
    axs[0].set_ylabel(r"$\rho(i)$")
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sampling time and network depth are nested progressions (Fig. 7, FLUX panels, "
                 "single-stream blocks)", y=1.03)
    save(fig, "fig7_flux_four_step_rho.png")

if exists("flux", "two_step_jay_decodes.pt"):
    two = torch.load(DATA / "flux/two_step_jay_decodes.pt")
    cols = list(range(0, 38, 3)) + [37]
    fig = plt.figure(figsize=(15, 2.9))
    gs = fig.add_gridspec(2, len(cols), wspace=0.03, hspace=0.25)
    for k in range(2):
        image_row([fig.add_subplot(gs[k, j]) for j in range(len(cols))], two["endpoints"][k][cols],
                  [f"sb{c:02d}" for c in cols] if k == 0 else [""] * len(cols))
        fig.text(0.115, 0.68 - 0.42 * k, f"step {k + 1}\n" + ("1 → 0.5" if k == 0 else "0.5 → 0"),
                 ha="right", fontsize=9)
    fig.suptitle(r"Two-step FLUX-schnell endpoint predictions $\tilde x_r^i$ per single block "
                 "(Fig. 6)", y=1.04)
    save(fig, "fig6_flux_two_step_endpoint_decodes.png")

(ROOT / "comparison.json").write_text(json.dumps(cmp, indent=1))


# ------------------------------------------------- FLUX sensitivity runs (data/flux_* variants)
def variant_stats(name: str) -> dict:
    d = DATA / name
    p_fx = paper_vec["coeffs_flux_sb"]
    res: dict = {}
    if (d / "mixing.json").exists():
        m = json.loads((d / "mixing.json").read_text())
        res["alpha_mean_rel_pct"] = coeff_stats(m["alpha"], p_fx["alpha"])["mean_relative_pct"]
        res["sigma_mean_rel_pct"] = coeff_stats(m["sigma"], p_fx["sigma"])["mean_relative_pct"]
        res["alpha_sb00"], res["sigma_sb00"] = m["alpha"][0], m["sigma"][0]
        res["mean_R"] = m["mean_held_out_r"]
    if (d / "multi_step_distances.pt").exists():
        ms = torch.load(d / "multi_step_distances.pt")
        for key, pk in (("endpoints", "fig7_flux_xr"), ("clean", "fig7_flux_x0")):
            ours = rho_multi(ms[key][:, NUM_DB:])
            paper = np.array(paper_ras[pk]["steps_sb38"])
            res[f"fig7_{key}_mae_pp"] = curve_stats(ours.ravel(), paper.ravel())["mae_pct_points"]
            res[f"fig7_{key}_per_step_pp"] = [curve_stats(ours[s], paper[s])["mae_pct_points"]
                                              for s in range(4)]
    if (d / "one_step_distances.pt").exists():
        one = torch.load(d / "one_step_distances.pt")
        pB = paper_ras["figB1_controls"]
        res["W_out_vs_fig3_pp"] = curve_stats(rho(one["model"]),
                                              paper_ras["fig3_rho"]["red"])["mae_pct_points"]
        res["ridge_vs_B1_blue_pp(swapped)"] = curve_stats(rho(one["ridge"]),
                                                          pB["blue"])["mae_pct_points"]
        res["random_vs_B1_grey_pp"] = curve_stats(rho(one["random"]), pB["grey"])["mae_pct_points"]
    return res


sensitivity = {n: variant_stats(n) for n in ("flux", "flux_512", "flux_altprompts", "flux_ridgefix")
               if (DATA / n).exists()}
save_json(sensitivity, ROOT / "sensitivity.json")


# -------------------------------------------------------------------- summary table (markdown)
def summary_table(c: dict) -> str:
    mf, fx = c["meanflow"], c.get("flux", {})
    mh = mf["mean_heldout_R"]
    rows = [
        "| Quantity | Paper | Ours | Difference |",
        "|---|---|---|---|",
        f"| MeanFlow alpha at block 4 | 2.18 (text) | {mf['alpha_block4']['ours']:.3f} "
        f"| {mf['alpha_block4']['diff_pct']:+.1f}% |",
        f"| MeanFlow sigma at block 4 | -6.04 (text) | {mf['sigma_block4']['ours']:.3f} "
        f"| {mf['sigma_block4']['diff_pct']:+.1f}% |",
        f"| MeanFlow mean held-out R (N=192, as Fig. B.3) | ~0.10 (text), "
        f"{mh['paper_fig_N192']:.3f} (fig) | {mh['ours_N192']:.3f} "
        f"| {mh['diff_pct_vs_fig']:+.1f}% |",
        f"| MeanFlow mean held-out R (N=256 run) | | {mh['ours_N256']:.3f} "
        f"| {mh['diff_pct_N256_vs_fig']:+.1f}% (other held-out split) |",
    ]
    for key, lab in (("alpha_curve", "MeanFlow alpha curve (12 blocks), Fig. 5"),
                     ("sigma_curve", "MeanFlow sigma curve (12 blocks), Fig. 5"),
                     ("heldout_R_curve", "MeanFlow held-out R curve (11 blocks, N=192), Fig. B.3")):
        st = mf[key]
        rows.append(f"| {lab} | digitized | | mean {st['mean_relative_pct']:.1f}%, max "
                    f"{st['max_relative_pct']:.1f}%, r={st['pearson_r']:.4f} |")
    ca = mf["centering_ablation"]
    rows.append(f"| Centering ablation: R change LayerNorm -> RMS-only | "
                f"{ca['paper_rms_vs_layernorm_change_pct']:+.2f}% | "
                f"{ca['ours_rms_vs_layernorm_change_pct']:+.2f}% | same conclusion |")
    if "alpha_curve" in fx:
        for key, lab in (("alpha_curve", "FLUX sb alpha curve (38 blocks), Fig. B.2"),
                         ("sigma_curve", "FLUX sb sigma curve (38 blocks), Fig. B.2")):
            st = fx[key]
            rows.append(f"| {lab} | digitized | | mean abs diff {st['mean_abs_diff']:.3f}, mean "
                        f"{st['mean_relative_pct']:.1f}%, r={st['pearson_r']:.3f} |")
        sb = fx["sigma_block0"]
        rows.append(f"| FLUX sigma at sb00 | {sb['paper_fig']:.3f} (fig) | {sb['ours']:.3f} "
                    f"| {sb['diff_pct']:+.1f}% |")
        rows.append(f"| FLUX mean held-out R (sb blocks) | not reported | "
                    f"{fx['mean_heldout_R']:.3f} | - |")
    labels = {
        "W_out_vs_paper_figB1_Wout_blue": "FLUX one-step rho, W_out vs Fig. B.1 W_out",
        "W_out_vs_paper_fig3_red": "FLUX one-step rho, W_out vs Fig. 3 curve",
        "ridge_vs_paper_figB1_trained_red": "FLUX one-step rho, trained probe vs Fig. B.1",
        "SWAPPED_W_out_vs_paper_figB1_red": "  same, legend swapped: W_out vs B.1 'trained'",
        "SWAPPED_ridge_vs_paper_figB1_blue": "  same, legend swapped: trained vs B.1 'W_out'",
        "random_vs_paper_figB1_random_grey": "FLUX one-step rho, random probe vs Fig. B.1",
        "W_out_pixel_space_vs_paper_figB1_Wout": "FLUX one-step rho, pixel-space W_out vs B.1",
    }
    for key, lab in labels.items():
        if key in fx.get("one_step_rho", {}):
            st = fx["one_step_rho"][key]
            rows.append(f"| {lab} | digitized | | MAE {st['mae_pct_points']:.1f} pp, max "
                        f"{st['max_abs_pct_points']:.1f} pp, r={st['pearson_r']:.3f} |")
    for key, lab in (("four_step_endpoints", "x_r, 38 sb blocks"),
                     ("four_step_clean", "x_0, 38 sb blocks"),
                     ("four_step_endpoints_all57_blocks", "x_r, all 57 blocks"),
                     ("four_step_clean_all57_blocks", "x_0, all 57 blocks")):
        if key in fx:
            st = fx[key]["all_steps"]
            per = ", ".join(f"s{k}: {fx[key][f'step{k}']['mae_pct_points']:.1f}"
                            for k in range(1, 5))
            rows.append(f"| FLUX 4-step rho ({lab}), Fig. 7 | digitized | | MAE "
                        f"{st['mae_pct_points']:.1f} pp ({per}), r={st['pearson_r']:.3f} |")
    return "\n".join(rows) + "\n"


(ROOT / "summary_table.md").write_text(summary_table(cmp))
print("wrote comparison.json, sensitivity.json, summary_table.md")
