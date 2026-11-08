# Phase 1 — PINN for potential flow past a cylinder

Unit cylinder, freestream 1, domain 1 ≤ r ≤ 5 with exact Dirichlet data on r = 5.
Network: 4 × 50 tanh MLP; 8192 interior collocation points (redrawn every 500 Adam steps).
Training: 5000 Adam steps (lr 0.001 → 0.0001), then up to 4000 L-BFGS iterations.
Mean ± standard deviation over seeds 0, 1, 2.

Velocity error: relative L2 norm of (u, v) on a 64 × 128 polar grid (geometric in r).
Steps: global count (Adam steps, then L-BFGS iterations after step 5000). Plateau: first step after which the best error improves by less than 1 % over any 500-step window.

| Case | Velocity error | ψ/φ error | max abs ΔCp | cl error | Adam time [s] | L-BFGS time [s] | Steps to 1e-2 | Steps to 1e-3 | Plateau step |
|---|---|---|---|---|---|---|---|---|---|
| ψ, soft BCs | 0.00119 ± 0.00015 | 0.000164 ± 2.2e-05 | 0.0235 ± 0.0055 | 0.000896 ± 0.00018 | 70.5 ± 1.4 | 97.1 ± 0.73 | 5600–5700 | never | 8800–9000 |
| ψ, hard BCs | 7.68e-05 ± 6.3e-06 | 1.7e-05 ± 3.7e-06 | 0.000443 ± 0.00018 | 0.000181 ± 0.00012 | 123 ± 2.5 | 60.3 ± 7.7 | 400–500 | 1300–1500 | 6400–6700 |
| ψ, hard BCs, Γ = 2π | 6.65e-05 ± 9.3e-06 | 1.36e-05 ± 2.2e-06 | 0.000493 ± 0.00011 | 0.000138 ± 5.4e-05 | 121 ± 0.87 | 72.7 ± 16 | 300–400 | 1000–1400 | 6500–7267 |
| φ, soft BCs (Neumann body) | 0.000193 ± 7.6e-06 | 4.66e-05 ± 1.2e-05 | 0.00168 ± 0.00017 | 7.39e-05 ± 6.1e-05 | 76.9 ± 0.63 | 100 ± 4.1 | 4600–5100 | 5700–5800 | 8650–9000 |

Device: NVIDIA GeForce RTX 2060 (clock capped at 1000 MHz). Training times exclude evaluation.

## Panel-method baseline (AERO-01)

Velocity errors here use the grid points with r ≥ 1.05 only: on the circle itself the panel solution is singular at the polygon nodes. On that subset the seed-0 PINNs score ψ, soft BCs 0.000913, ψ, hard BCs 8.49e-05, ψ, hard BCs, Γ = 2π 6.32e-05, φ, soft BCs (Neumann body) 0.000187.

| Flow | Panels | Velocity error | max abs ΔCp (control points) | Solve time [ms] |
|---|---|---|---|---|
| non-lifting | 64 | 0.00694 | 3.11e-15 | 0.84 |
| non-lifting | 128 | 0.0036 | 5.77e-15 | 3.6 |
| non-lifting | 256 | 0.00183 | 1.55e-14 | 12.6 |
| non-lifting | 512 | 0.000926 | 2.93e-14 | 52.5 |
| Γ = 2π (imposed) | 64 | 0.00621 | 0.0601 | 1.27 |
| Γ = 2π (imposed) | 128 | 0.00323 | 0.0313 | 3.59 |
| Γ = 2π (imposed) | 256 | 0.00164 | 0.0159 | 12.6 |
| Γ = 2π (imposed) | 512 | 0.000829 | 0.00805 | 52.7 |
