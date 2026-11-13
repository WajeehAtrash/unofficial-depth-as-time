# Paper values used for the comparison

These two files hold the numbers read out of the paper (arXiv 2610.03626, figures from its
LaTeX source). The comparison scripts read only these files, so the paper itself is not needed
to rerun them.

## How they were obtained

- **`paper_vector_data.json`**: the coefficient and held-out R plots are vector graphics. The
  plotted points were read from the path coordinates in the figure PDFs, and each axis was
  calibrated by a least-squares fit of the tick values against the tick-label positions.
- **`paper_raster_data.json`**: the rho plots (Fig. 3, Fig. B.1, Fig. 7) are bitmaps inside the
  figure PDFs. Each curve was isolated by colour and read column by column, with the axes
  calibrated from the major tick marks. Fig. B.1 and Fig. 3 are read at the 57 block positions
  and Fig. 7 at 57 and at the 38 single-stream positions per step.
- **Check:** the same procedure reproduces the numbers the paper prints in its text. For
  MeanFlow it reads alpha = 2.17 and sigma = -6.045 at block 4 (the text says 2.18 and -6.04),
  and 0.995 / -0.005 at the last block (exactly 1 / 0 by construction). For the bitmap plots the
  axis calibration fits its tick marks to within 0.0003 (stored as `calibration_residual`).

The extraction scripts were removed from the repository once the values were stored; they are
in the git history (`extract_paper_vector.py`, `extract_paper_raster.py`).

## Contents

`paper_vector_data.json`

| Key | Content |
|---|---|
| `coeffs_meanflow` | alpha and sigma per block, MeanFlow (12 blocks), from Fig. B.2 |
| `coeffs_meanflow_fig5` | the same from Fig. 5 (agrees with the line above) |
| `coeffs_flux_sb` | alpha and sigma per block, FLUX single-stream blocks (38), Fig. B.2 |
| `heldout_R_meanflow_layernorm` | held-out R per block (1 to 11), MeanFlow, Fig. B.3 |
| `heldout_R_meanflow_rmsonly` | the same with RMS-only normalization (1 to 12) |

`paper_raster_data.json`

| Key | Content |
|---|---|
| `fig3_rho.red` | FLUX one-step rho at the 57 blocks, Fig. 3 |
| `figB1_controls.{red,blue,grey}` | the three FLUX curves of Fig. B.1 (57 blocks), by colour |
| `fig7_flux_xr`, `fig7_flux_x0` | FLUX four-step rho, Fig. 7: `steps` is 4 x 57, `steps_sb38` is 4 x 38 |

`calibration_residual` is the largest tick-fit error of that figure, in axis units.
