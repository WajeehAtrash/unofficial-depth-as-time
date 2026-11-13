| Quantity | Paper | Ours | Difference |
|---|---|---|---|
| MeanFlow alpha at block 4 | 2.18 (text) | 2.197 | +0.8% |
| MeanFlow sigma at block 4 | -6.04 (text) | -6.132 | -1.5% |
| MeanFlow mean held-out R (N=192, as Fig. B.3) | ~0.10 (text), 0.099 (fig) | 0.100 | +1.0% |
| MeanFlow mean held-out R (N=256 run) | | 0.094 | -4.3% (other held-out split) |
| MeanFlow alpha curve (12 blocks), Fig. 5 | digitized | | mean 1.5%, max 2.6%, r=0.9998 |
| MeanFlow sigma curve (12 blocks), Fig. 5 | digitized | | mean 1.1%, max 2.0%, r=1.0000 |
| MeanFlow held-out R curve (11 blocks, N=192), Fig. B.3 | digitized | | mean 1.3%, max 3.4%, r=0.9970 |
| Centering ablation: R change LayerNorm -> RMS-only | -0.17% | -0.19% | same conclusion |
| FLUX sb alpha curve (38 blocks), Fig. B.2 | digitized | | mean abs diff 0.036, mean 9.1%, r=0.985 |
| FLUX sb sigma curve (38 blocks), Fig. B.2 | digitized | | mean abs diff 0.576, mean 32.9%, r=1.000 |
| FLUX sigma at sb00 | -5.842 (fig) | -4.639 | +20.6% |
| FLUX mean held-out R (sb blocks) | not reported | 0.131 | - |
| FLUX sb alpha curve, head conditioned as in the paper | digitized | | mean abs diff 0.037, mean 8.1%, r=0.989 |
| FLUX sb sigma curve, head conditioned as in the paper | digitized | | mean abs diff 0.077, mean 3.4%, r=1.000 |
|   same: sigma at sb00 | -5.842 (fig) | -6.104 | -4.5% |
|   same: last block (alpha, sigma) | (0.964, -0.161) | (0.943, -0.159) | |
| FLUX one-step rho, W_out vs Fig. B.1 W_out | digitized | | MAE 4.0 pp, max 9.6 pp, r=0.997 |
| FLUX one-step rho, W_out vs Fig. 3 curve | digitized | | MAE 1.1 pp, max 3.3 pp, r=1.000 |
| FLUX one-step rho, trained probe vs Fig. B.1 | digitized | | MAE 2.8 pp, max 17.7 pp, r=0.996 |
|   W_out with head conditioned as in the paper vs Fig. B.1 W_out | digitized | | MAE 1.0 pp, max 3.5 pp, r=1.000 |
| FLUX one-step rho, W_out vs Fig. B.1 trained probe | digitized | | MAE 1.1 pp, max 3.3 pp, r=1.000 |
| FLUX one-step rho, random probe vs Fig. B.1 | digitized | | MAE 3.0 pp, max 10.4 pp, r=0.977 |
| FLUX one-step rho, pixel-space W_out vs B.1 | digitized | | MAE 24.5 pp, max 50.8 pp, r=0.820 |
| FLUX 4-step rho (x_r, 38 sb blocks), Fig. 7 | digitized | | MAE 4.1 pp (s1: 8.5, s2: 4.2, s3: 1.9, s4: 1.8), r=0.975 |
| FLUX 4-step rho (x_r, head as in the paper), Fig. 7 | digitized | | MAE 2.5 pp (s1: 1.1, s2: 1.7, s3: 2.8, s4: 4.5), r=0.993 |
| FLUX 4-step rho (x_0, 38 sb blocks), Fig. 7 | digitized | | MAE 2.0 pp (s1: 1.3, s2: 2.6, s3: 2.4, s4: 1.5), r=0.992 |
| FLUX 4-step rho (x_0, head as in the paper), Fig. 7 | digitized | | MAE 0.8 pp (s1: 0.8, s2: 0.9, s3: 0.9, s4: 0.7), r=0.999 |
| FLUX 4-step rho (x_r, all 57 blocks), Fig. 7 | digitized | | MAE 14.0 pp (s1: 15.1, s2: 15.2, s3: 13.9, s4: 11.8), r=0.764 |
| FLUX 4-step rho (x_0, all 57 blocks), Fig. 7 | digitized | | MAE 9.2 pp (s1: 9.6, s2: 14.1, s3: 9.0, s4: 4.0), r=0.909 |
