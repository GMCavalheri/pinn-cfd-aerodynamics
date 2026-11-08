# Phase 5 — Exploratory: PINNs for the steady 2D Euler equations

Conservation-form residuals, primitive outputs (density and pressure through a softplus), 5 × 64 tanh network, 8192 interior points. Training: 6000 Adam steps (lr 0.001 → 5e-05), then up to 2000 L-BFGS iterations.

State error: relative L2 norm of (ρ, u, v, p) against the exact solution (Ringleb: 4096 area-uniform points; wedge: 101 × 101 grid). Wedge: 'smooth' excludes a band of ±0.05 around the exact shock; the shock angle is where the predicted density crosses its mid value on x = 0.4, 0.6, 0.8, 1.0 (exact β = 39.31°).

| Case | Seeds | State error | ρ error | p error | Smooth-region error | β [deg] | Time [s] | Steps to 1e-1 | Steps to 3e-2 | Steps to 1e-2 | Plateau step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ringleb flow (smooth, transonic) | 2 | 0.00143 ± 2.2e-06 | 0.000958 ± 0.00016 | 0.000607 ± 0.0001 | — | — | 421 ± 1.2 | 200 | 600–700 | 2900–4000 | 7900–8000 |
| wedge, plain PINN | 1 | 0.0531 | 0.0625 | 0.0916 | 0.0301 | 37.5 | 448 | 5400 | never | never | 6900 |
| wedge, artificial viscosity ν = 0.01 | 1 | 0.0456 | 0.0555 | 0.0776 | 0.0225 | 38.4 | 1.34e+03 | 4100 | never | never | 7000 |
| wedge, residual down-weighted at the shock | 1 | 0.056 | 0.0655 | 0.0948 | 0.0277 | 36.9 | 580 | 6100 | never | never | 6800 |

Device: NVIDIA GeForce RTX 2060 (clock capped at 1000 MHz). Training times exclude evaluation.
