# Phase 2 — PINN for the flow around a NACA 4-digit family

Family: m ∈ [0, 0.06], p ∈ [0.3, 0.5], t ∈ [0.08, 0.18], α ∈ [−4°, 8°]. One network
ψ(x, y; m, p, t, α) for the whole family: freestream + vortex + learned disturbance, the
network seeing inverse-Joukowski (circle-plane) coordinates; see
`src/pinncfd/pinn_airfoil_flowfield/model.py`. Lift: circulation of the predicted
velocity on r = 1.5 c (Kutta–Joukowski) and surface-pressure integration.
Reference: AERO-01 Hess-Smith panel method with 400 panels.

Field network 5 × 64 tanh, case head 2 × 32. Per Adam step: 32 random shapes × 384 interior points (+ surface, Kutta, far-field points), panel data from 16 of the 40 training cases × 128 points; redrawn every 250 steps.
Training: 4000 Adam steps (lr 0.001 → 5e-05), then up to 1000 L-BFGS iterations on the last batch.

Velocity error: relative L2 norm of (u, v) at 512 field points per case (no closer than 0.02 c to the surface), averaged over cases. Validation: 8 cases (used during training to monitor convergence). Test: 6 sections × 4 angles, never used before the final evaluation. Plateau window 1000 steps, tolerance 1 %; a plateau at the last step (5000) means the run was still improving when the budget ran out.

## Training runs (validation set)

| Run | Seeds | Val velocity error | Val Cp RMS | Val cl error | Adam time [s] | L-BFGS time [s] | Steps to 1e-1 | Steps to 3e-2 | Steps to 1e-2 | Plateau step |
|---|---|---|---|---|---|---|---|---|---|---|
| physics + panel data | 3 | 0.0117 ± 0.003 | 0.124 ± 0.013 | 0.0843 ± 0.034 | 446 ± 25 | 129 ± 6.9 | 250–750 | 3000–4500 | 5000 (1/3 seeds) | 5000 |
| physics only | 1 | 0.0204 | 0.119 | 0.182 | 355 | 108 | 3250 | 4750 | never | 5000 |
| panel data only | 1 | 0.00619 | 0.294 | 0.022 | 62.5 | 26.6 | 250 | 750 | 4250 | 5000 |
| physics + data, plain coordinates | 1 | 0.0253 | 0.265 | 0.34 | 278 | 83.1 | 1000 | 4750 | never | 5000 |

Device: NVIDIA GeForce RTX 2060 (clock capped at 1000 MHz).

## Held-out test sections (seed-0 networks)

| Run | Velocity error | Cp RMS | Cp RMS, 0.05 < x < 0.95 | max abs ΔCp | cl error (circulation) | cl error (pressure) |
|---|---|---|---|---|---|---|
| physics + panel data | 0.0116 | 0.116 | 0.0513 | 1.29 | 0.121 | 0.0554 |
| physics only | 0.0211 | 0.12 | 0.0767 | 0.873 | 0.181 | 0.11 |
| panel data only | 0.00876 | 0.325 | 0.0306 | 5.14 | 0.0351 | 0.0403 |
| physics + data, plain coordinates | 0.0254 | 0.301 | 0.177 | 1.35 | 0.331 | 0.0634 |

Per test case, physics + panel data (seed 0):

| Case | Velocity error | Cp RMS | cl panel | cl PINN (circulation) | cl PINN (pressure) |
|---|---|---|---|---|---|
| NACA 0012 α=-3.0° | 0.0095 | 0.1 | -0.3627 | -0.2831 | -0.3030 |
| NACA 0012 α=+1.0° | 0.00824 | 0.0699 | 0.1209 | 0.0996 | 0.1413 |
| NACA 0012 α=+4.0° | 0.00906 | 0.0757 | 0.4834 | 0.3995 | 0.4793 |
| NACA 0012 α=+7.0° | 0.0172 | 0.134 | 0.8445 | 0.6785 | 0.7800 |
| NACA 1408 α=-3.0° | 0.0133 | 0.0932 | -0.2273 | -0.1930 | -0.1464 |
| NACA 1408 α=+1.0° | 0.00571 | 0.0523 | 0.2416 | 0.1702 | 0.2785 |
| NACA 1408 α=+4.0° | 0.00713 | 0.104 | 0.5926 | 0.4578 | 0.5964 |
| NACA 1408 α=+7.0° | 0.0184 | 0.226 | 0.9419 | 0.7303 | 0.8836 |
| NACA 2412 α=-3.0° | 0.00377 | 0.0529 | -0.1033 | -0.1111 | -0.0998 |
| NACA 2412 α=+1.0° | 0.00443 | 0.0636 | 0.3806 | 0.3047 | 0.3897 |
| NACA 2412 α=+4.0° | 0.00539 | 0.09 | 0.7424 | 0.6270 | 0.7549 |
| NACA 2412 α=+7.0° | 0.00853 | 0.119 | 1.1022 | 0.9229 | 1.0754 |
| NACA 3315 α=-3.0° | 0.0155 | 0.0996 | 0.0027 | -0.0386 | -0.0519 |
| NACA 3315 α=+1.0° | 0.0136 | 0.0981 | 0.4980 | 0.3715 | 0.4307 |
| NACA 3315 α=+4.0° | 0.0121 | 0.0885 | 0.8680 | 0.6980 | 0.7960 |
| NACA 3315 α=+7.0° | 0.0167 | 0.113 | 1.2356 | 1.0132 | 1.1234 |
| NACA 4415 α=-3.0° | 0.00745 | 0.163 | 0.1634 | 0.1277 | 0.1425 |
| NACA 4415 α=+1.0° | 0.00495 | 0.11 | 0.6587 | 0.5659 | 0.6494 |
| NACA 4415 α=+4.0° | 0.00644 | 0.0982 | 1.0282 | 0.8959 | 1.0147 |
| NACA 4415 α=+7.0° | 0.00976 | 0.115 | 1.3948 | 1.1945 | 1.3248 |
| NACA 5518 α=-3.0° | 0.0131 | 0.174 | 0.3707 | 0.3366 | 0.3377 |
| NACA 5518 α=+1.0° | 0.0107 | 0.127 | 0.8768 | 0.7530 | 0.7979 |
| NACA 5518 α=+4.0° | 0.0187 | 0.151 | 1.2538 | 1.0442 | 1.1057 |
| NACA 5518 α=+7.0° | 0.038 | 0.257 | 1.6273 | 1.3025 | 1.3585 |

## Outside the family (seed 0, physics + panel data)

| Case | Leaves the box in | Velocity error | Cp RMS | cl panel | cl PINN (circulation) |
|---|---|---|---|---|---|
| NACA 0021 α=+4.0° | t | 0.0113 | 0.141 | 0.5172 | 0.5575 |
| NACA 2412 α=+12.0° | α | 0.0383 | 0.373 | 1.6947 | 1.3275 |
| NACA 8412 α=+4.0° | m | 0.0385 | 0.266 | 1.5087 | 1.2123 |
| NACA 2406 α=+4.0° | t | 0.0106 | 0.163 | 0.7037 | 0.5378 |

## The reference and the baseline

Panel method with 200 panels (the baseline): 7.77 ms per case (contour + assemble + solve, median, NumPy with its default BLAS threads).

| Panels | Velocity difference to N = 800 | cl difference to N = 800 |
|---|---|---|
| 200 | 0.000706 | 0.00101 |
| 400 | 0.000241 | 0.000358 |

Joukowski airfoil (mx = 0.1, my = 0.05, α = 4°), 400 panels: cl within 1.53 % of the exact value.
