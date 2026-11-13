"""Digitize the raster rho(i) plots of the paper (Fig. 3 right, Fig. B.1, Fig. 7 FLUX panels).

The plots are bitmaps inside the figure PDFs; they are extracted to ``paper_data/raster/*.png``
(gitignored) on first use from the arXiv source in ``paper/latex/figs``.
Axes are calibrated from the major tick marks on the left and bottom spines (0.0 ... 1.0 in
steps of 0.2 on both axes). Each curve is isolated by colour and read column by column.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from common import ROOT

RAW = ROOT / "paper_data/raster"
FIGS = ROOT.parent / "paper/latex/figs"  # arXiv e-print source, extracted by the user
SOURCES = {  # bitmap name -> figure PDF that embeds it
    "fig3_rho": "second_examples_trimmed.pdf",
    "figB1_controls": "qualitative_examples_trimmed.pdf",
    "fig7_multistep": "distillation_nested_trimmed.pdf",
}


def ensure_bitmaps() -> None:
    """Extract the plot bitmaps from the figure PDFs (the largest image on page 1) if missing."""
    import fitz  # PyMuPDF, only needed for this step

    RAW.mkdir(parents=True, exist_ok=True)
    for name, pdf in SOURCES.items():
        target = RAW / f"{name}.png"
        if target.exists():
            continue
        doc = fitz.open(FIGS / pdf)
        images = doc[0].get_images(full=True)
        xref = max(images, key=lambda info: info[2] * info[3])[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.alpha or pix.n > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        pix.save(str(target))


def load(name):
    return np.asarray(Image.open(RAW / f"{name}.png").convert("RGB")).astype(int)


def is_black(a):
    return (a < 40).all(axis=-1)


def runs(mask_1d):
    """Return (start, length) of consecutive True runs."""
    m = np.concatenate([[0], mask_1d.astype(int), [0]])
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(zip(starts, ends - starts))


def find_spines(a, x_lo=0, x_hi=None):
    """Left spine column and bottom spine row of the panel in columns [x_lo, x_hi)."""
    x_hi = a.shape[1] if x_hi is None else x_hi
    m = 15  # skip a possible black border around the bitmap
    x_lo = max(x_lo, m)
    x_hi = min(x_hi, a.shape[1] - m)
    blk = is_black(a[m:-m, x_lo:x_hi])
    def longest_run(mask_1d):
        return max([length for _, length in runs(mask_1d)] or [0])

    col_best = max(range(blk.shape[1]), key=lambda c: longest_run(blk[:, c]))
    row_best = max(range(blk.shape[0]), key=lambda r: longest_run(blk[r]))
    return x_lo + col_best, m + row_best


def major_ticks_y(a, spine_x, bottom, n=6):
    """Rows of the n longest tick marks left of the spine (bottom-up order)."""
    left = spine_x
    while is_black(a[bottom - 400 : bottom - 10, left - 1]).mean() > 0.5:  # spine thickness
        left -= 1
    blk = is_black(a[: bottom - 3, left - 60 : left - 1])
    blk[:15] = False  # bitmap border
    lengths = blk.sum(axis=1)
    rows = [r for r in range(len(lengths)) if lengths[r] > 0]
    groups = []
    for r in rows:
        if groups and r - groups[-1][-1] <= 1:
            groups[-1].append(r)
        else:
            groups.append([r])
    groups = sorted(groups, key=lambda g: -max(lengths[g]))
    # keep groups whose length is close to the maximum (major ticks), drop label glyph clutter
    cand = [g for g in groups if max(lengths[g]) >= 0.8 * max(lengths[groups[0]]) and len(g) < 20]
    centers = sorted([np.mean(g) for g in cand], reverse=True)
    assert len(centers) >= n, centers
    return centers[:n]


def major_ticks_x(a, spine_x, bottom, x_hi, n=6):
    x_hi = min(x_hi, a.shape[1] - 15)
    blk_all = is_black(a[:, spine_x:x_hi])
    below = bottom
    while blk_all[below].mean() > 0.5:  # skip the thickness of the spine
        below += 1
    blk = is_black(a[below + 1 : below + 60, spine_x + 15 : x_hi])
    spine_x = spine_x + 15
    lengths = blk.sum(axis=0)
    cols = [c for c in range(len(lengths)) if lengths[c] > 0]
    groups = []
    for c in cols:
        if groups and c - groups[-1][-1] <= 1:
            groups[-1].append(c)
        else:
            groups.append([c])
    top = max(max(lengths[g]) for g in groups)
    cand = [g for g in groups if max(lengths[g]) >= 0.8 * top and len(g) < 20]
    centers = sorted(spine_x + np.mean(g) for g in cand)
    assert len(centers) >= n, centers
    return centers[:n]


def calibration(a, x_lo=0, x_hi=None):
    x_hi = a.shape[1] if x_hi is None else x_hi
    sx, by = find_spines(a, x_lo, x_hi)
    ys = major_ticks_y(a, sx, by)
    xs = major_ticks_x(a, sx, by, x_hi)
    vals = np.linspace(0, 1, 6)
    ay, by_ = np.polyfit(ys, vals, 1)
    ax, bx = np.polyfit(xs, vals, 1)
    px_to_y = lambda r: ay * np.asarray(r) + by_  # noqa: E731
    x_to_px = lambda x: (np.asarray(x) - bx) / ax  # noqa: E731
    resid = max(np.abs(np.polyval([ay, by_], ys) - vals).max(),
                np.abs(np.polyval([ax, bx], xs) - vals).max())
    return px_to_y, x_to_px, sx, by, resid


COLOURS = {
    "red": lambda a: (a[..., 0] > 140) & (a[..., 1] < 70) & (a[..., 2] < 85),
    "blue": lambda a: (a[..., 2] > 150) & (a[..., 0] < 110) & (a[..., 1] > 90) & (a[..., 1] < 150),
    "dark": lambda a: (np.abs(a - 62).max(axis=-1) < 14),
    "grey": lambda a: (np.abs(a - 132).max(axis=-1) < 14),
}


def sample_curve(a, colour, xs_px, px_to_y, top, bottom, half_window=3, side="both"):
    mask = COLOURS[colour](a)
    out = []
    for x in xs_px:
        x = int(round(x))
        lo, hi = (x - half_window, x + half_window + 1)
        if side == "right":
            lo, hi = x, x + 2 * half_window + 1
        elif side == "left":
            lo, hi = x - 2 * half_window, x + 1
        rows = np.where(mask[top:bottom, lo:hi].any(axis=1))[0] + top
        if not len(rows):
            out.append(np.nan)
        elif side == "right":  # step start: the curve peaks there, take the top of the line
            out.append(float(px_to_y(rows.min() + 4)))
        else:
            out.append(float(px_to_y(np.median(rows))))
    out = np.array(out)
    # fill gaps (dashes, crossings) by linear interpolation along the curve
    ok = ~np.isnan(out)
    if ok.sum() >= 2 and (~ok).any():
        idx = np.arange(len(out))
        out[~ok] = np.interp(idx[~ok], idx[ok], out[ok])
    return out


def main():
    ensure_bitmaps()
    data = {}
    L = 57

    for name, curves in (("fig3_rho", ("red",)), ("figB1_controls", ("red", "blue", "grey"))):
        a = load(name)
        px_to_y, x_to_px, sx, by, resid = calibration(a)
        xs = x_to_px(np.arange(L) / (L - 1))
        res = {"calibration_residual": resid}
        for c in curves:
            res[c] = sample_curve(a, c, xs, px_to_y, 0, by).tolist()
        data[name] = res

    # Fig. 7: four panels; FLUX x_r is panel 0, FLUX x_0 is panel 1
    a = load("fig7_multistep")
    width = a.shape[1]
    bounds = [(0, int(width * 0.26)), (int(width * 0.25), int(width * 0.51))]
    for key, (lo, hi) in zip(("flux_xr", "flux_x0"), bounds):
        px_to_y, x_to_px, sx, by, resid = calibration(a, lo, hi)
        entry = {"calibration_residual": resid}
        # read at the positions of all 57 blocks and of the 38 single-stream blocks
        for label, n in (("steps", L), ("steps_sb38", 38)):
            steps = []
            for s in range(4):
                xs = x_to_px((s + np.arange(n) / n) / 4)
                curve = sample_curve(a, "red", xs, px_to_y, 0, by, half_window=2)
                # the first point of a step sits on the step boundary: read the peak to its right
                curve[0] = sample_curve(a, "red", xs[:1], px_to_y, 0, by, half_window=2,
                                        side="right")[0]
                steps.append(curve.tolist())
            entry[label] = steps
        data[f"fig7_{key}"] = entry

    (ROOT / "paper_data/paper_raster_data.json").write_text(json.dumps(data, indent=1))
    for k, v in data.items():
        print(k, "calib resid", round(v["calibration_residual"], 4))
        for c, vals in v.items():
            if c == "calibration_residual":
                continue
            arr = np.array(vals)
            print("  ", c, np.round(arr if arr.ndim == 1 else arr[:, [0, 10, 20, 30, -1]], 3))


if __name__ == "__main__":
    main()
