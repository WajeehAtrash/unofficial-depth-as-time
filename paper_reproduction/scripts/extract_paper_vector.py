"""Digitize the vector plots of the paper (coefficients and held-out R) from the arXiv figure PDFs.

Tick labels are text, so each axis is calibrated by a least-squares fit of tick value against tick
label centre. Minus signs are drawn as paths, so tick signs are supplied here by hand.
Polyline vertices give the plotted values exactly.
"""
import json
import sys
from pathlib import Path

import fitz
import numpy as np

FIGS = Path(__file__).resolve().parents[2] / "paper/latex/figs"
OUT = Path(__file__).resolve().parents[1] / "paper_data"
TEAL = (0.0, 0.306, 0.392)    # alpha
RED = (0.698, 0.125, 0.157)   # sigma
BLUE = (0.122, 0.467, 0.706)  # matplotlib C0
ORANGE = (1.0, 0.498, 0.055)  # matplotlib C1


def words(page):
    return [((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, w[4]) for w in page.get_text("words")]


def calib(pairs):
    """Least-squares map pixel -> value from (pixel, value) pairs."""
    p, v = np.array(pairs, dtype=float).T
    a, b = np.polyfit(p, v, 1)
    return lambda x: a * np.asarray(x) + b


def tick_pairs(ws, xr, yr, values, axis):
    """Pick tick labels inside a box (x range, y range) in reading order and pair with values."""
    sel = [w for w in ws if xr[0] <= w[0] <= xr[1] and yr[0] <= w[1] <= yr[1]]
    key = 0 if axis == "x" else 1
    sel.sort(key=lambda w: w[key])
    assert len(sel) == len(values), (axis, [s[2] for s in sel], values)
    return [(w[key], v) for w, v in zip(sel, values)]


def polylines(page, color, n_items=None):
    out = []
    for d in page.get_drawings():
        col = d.get("color")
        if col is None or d.get("fill") is not None:
            continue
        if max(abs(a - b) for a, b in zip(col, color)) > 0.01:
            continue
        pts = [it[1] for it in d["items"] if it[0] == "l"]
        if d["items"] and d["items"][-1][0] == "l":
            pts.append(d["items"][-1][2])
        if n_items is None or len(d["items"]) == n_items:
            out.append(np.array([(p.x, p.y) for p in pts]))
    return out


def in_x(arr, lo, hi):
    return [a for a in arr if lo <= a[:, 0].mean() <= hi]


def main():
    data = {}

    # ---- all_coeffs_5panel: MeanFlow (panel 1) and FLUX sb blocks (panel 3)
    page = fitz.open(FIGS / "all_coeffs_5panel.pdf")[0]
    ws = words(page)
    panels = {
        "meanflow": dict(xr=(80, 460), yt=[-7.5, -5.0, -2.5, 0.0, 2.5],
                         xt=[0, 2.5, 5, 7.5, 10], ytx=(55, 75), n=11),
        "flux_sb": dict(xr=(930, 1300), yt=[-6, -4, -2, 0], xt=[0, 10, 20, 30],
                        ytx=(915, 928), n=37),
    }
    for name, p in panels.items():
        fx = calib(tick_pairs(ws, (p["xr"][0], p["xr"][1]), (245, 256), p["xt"], "x"))
        fy = calib(tick_pairs(ws, p["ytx"], (20, 240), sorted(p["yt"], reverse=True), "y"))
        res = {}
        for key, col in (("alpha", TEAL), ("sigma", RED)):
            lines = in_x(polylines(page, col, p["n"]), *p["xr"])
            assert len(lines) == 1, (name, key, len(lines))
            line = lines[0]
            res[key] = fy(line[:, 1]).tolist()
            res["block0"] = np.round(fx(line[:, 0]), 2).tolist()
        data[f"coeffs_{name}"] = res

    # ---- meanflow_drifting_coeffs (Fig. 5), MeanFlow panel, as a cross-check of the 5-panel plot
    page = fitz.open(FIGS / "meanflow_drifting_coeffs.pdf")[0]
    ws = words(page)
    fx = calib(tick_pairs(ws, (90, 470), (252, 264), [0, 2, 4, 6, 8, 10], "x"))
    fy = calib(tick_pairs(ws, (55, 75), (20, 245), [2.5, 0.0, -2.5, -5.0, -7.5], "y"))
    res = {}
    for key, col in (("alpha", TEAL), ("sigma", RED)):
        lines = in_x(polylines(page, col, 11), 60, 480)
        assert len(lines) == 1
        res[key] = fy(lines[0][:, 1]).tolist()
        res["block0"] = np.round(fx(lines[0][:, 0]), 2).tolist()
    data["coeffs_meanflow_fig5"] = res

    # ---- centering_ablation_R, MeanFlow panel: held-out R per block (1-based, 1..12)
    page = fitz.open(FIGS / "centering_ablation_R.pdf")[0]
    ws = words(page)
    fx = calib(tick_pairs(ws, (60, 360), (252, 262), [2, 4, 6, 8, 10, 12], "x"))
    fy = calib(tick_pairs(ws, (25, 40), (30, 215), [1.0, 0.8, 0.6, 0.4, 0.2], "y"))
    line = in_x(polylines(page, BLUE), 60, 360)
    line = max(line, key=len)
    data["heldout_R_meanflow_layernorm"] = {
        "block1": np.round(fx(line[:, 0]), 2).tolist(),
        "R": fy(line[:, 1]).tolist(),
    }
    oline = max(in_x(polylines(page, ORANGE), 60, 360), key=len)
    data["heldout_R_meanflow_rmsonly"] = {
        "block1": np.round(fx(oline[:, 0]), 2).tolist(),
        "R": fy(oline[:, 1]).tolist(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paper_vector_data.json").write_text(json.dumps(data, indent=1))
    for k, v in data.items():
        print(k, {kk: np.round(vv, 3).tolist() for kk, vv in v.items()})


if __name__ == "__main__":
    sys.exit(main())
