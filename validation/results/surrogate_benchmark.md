# Phase 4 — The PINN as a surrogate for the panel method

Surrogate: Phase 3 champion, seed 0 (6851 parameters; validation velocity error 0.00588; training time 657 s on the RTX 2060 at 1000 MHz, after a 40-minute architecture search). Chosen on validation error. Baseline: AERO-01 panel method, 200 panels. Reference for accuracy: 400 panels.

## Lift accuracy on 500 random sections of the family

| Method | Mean abs cl error | 95th percentile | Max |
|---|---|---|---|
| panel method, 200 panels | 0.000538 | 0.00123 | 0.0015 |
| PINN, surface pressure (120 points) | 0.0314 | 0.0804 | 0.137 |
| PINN, circulation (256 points on r = 1.5c) | 0.0351 | 0.0882 | 0.121 |

## Wall time for n lift evaluations

| n | Panel method (CPU) | PINN, GPU (pressure) | PINN, GPU (circulation) | PINN, CPU (pressure) | Contours alone (CPU) | Speed-up, GPU |
|---|---|---|---|---|---|---|
| 1 | 0.00794 s | 0.00886 s | 0.00871 s | 0.00373 s | 0.000161 s | 0.896× |
| 10 | 0.0887 s | 0.0109 s | 0.00891 s | 0.0107 s | 0.00177 s | 8.17× |
| 100 | 0.811 s | 0.035 s | 0.00843 s | 0.0395 s | 0.013 s | 23.2× |
| 1000 | 8.16 s | 0.157 s | 0.071 s | 0.366 s | 0.13 s | 51.8× |
| 10000 | 79.9 s | 1.54 s | 0.658 s | 2.78 s | 1.31 s | 52× |

Per case at n = 10000: panel 7.99 ms, PINN GPU 154 µs, PINN CPU 278 µs. Panel times above 1000 cases are the 1000-case time scaled (the method is strictly sequential). The pressure-lift PINN times include building each section's 120-panel contour in NumPy (column 'Contours alone'); the circulation lift needs no contour. GPU times are synchronised; all times are medians after a warm-up call.

**Break-even** (training time charged to the PINN): 83,739 evaluations on the GPU, 85,088 on the CPU; 390,638 on the GPU if the architecture search is charged too.

## The equal-accuracy baseline

The PINN's lift is less accurate than the 200-panel baseline, so the fair comparison is with the coarsest panel method that is at least as accurate. Panel count sweep on the same 500 sections (error against 400 panels; time per case over 1000 other sections, one at a time):

| Panels | Mean abs cl error | 95th percentile | Time per case |
|---|---|---|---|
| 8 | 0.0749 | 0.15 | 0.338 ms |
| 12 | 0.0356 | 0.0755 | 0.37 ms |
| 16 | 0.0208 | 0.0462 | 0.388 ms |
| 24 | 0.0099 | 0.0238 | 0.461 ms |
| 32 | 0.00616 | 0.0147 | 0.515 ms |
| 48 | 0.00353 | 0.00774 | 0.715 ms |
| 64 | 0.00252 | 0.00528 | 0.946 ms |
| 100 | 0.0015 | 0.00326 | 2.61 ms |
| 200 | 0.000538 | 0.00123 | 8.17 ms |

The coarsest panel method at least as accurate as the PINN's pressure lift (mean 0.0314) has **16 panels**: 0.388 ms per case, against the PINN's 154 µs (pressure, GPU, contours included) and 65.8 µs (circulation, GPU) at n = 10000: 2.52× and 5.89× faster per case. **Break-even against it**: 2,803,732 evaluations (pressure) or 2,038,998 (circulation), training time charged, search not.

## Design loop (a): the incidence for cl = 0.8

Panel: Brent's method on [−10°, 15°], tolerance 1e-6 rad, one section at a time (0.266 s for all 6). PINN, Newton: all 6 sections at once from α = 0, d cl/dα by autograd, 4 batched network evaluations, 0.0636 s. PINN, sweep: the whole cl–α curve of every section (321 incidences, 0.29 s), then interpolation. The PINN's answer is checked by a panel solve at its incidence.

| Section | α panel [deg] | Panel solves | Panel time [s] | α PINN, Newton [deg] | α PINN, sweep [deg] | cl (panel) at α PINN |
|---|---|---|---|---|---|---|
| NACA 0012 | 6.620 | 6 | 0.0423 | 6.673 | 6.673 | 0.806 |
| NACA 1408 | 5.771 | 6 | 0.0442 | 5.936 | 5.936 | 0.819 |
| NACA 2412 | 4.473 | 6 | 0.0442 | 4.660 | 4.660 | 0.823 |
| NACA 3315 | 3.442 | 6 | 0.0443 | 3.500 | 3.500 | 0.807 |
| NACA 4415 | 2.142 | 6 | 0.0441 | 2.318 | 2.318 | 0.822 |
| NACA 5518 | 0.391 | 6 | 0.047 | 0.796 | 0.796 | 0.851 |

## Design loop (b): a cl carpet over camber and thickness

41 × 41 sections (m 0–6 %, t 8–18 %, p = 0.4, α = 4°): panel 13.7 s, PINN 0.26 s (52.7× faster); cl difference mean 0.0298, max 0.082.
