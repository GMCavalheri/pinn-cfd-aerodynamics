# Phase 3 — Neuroevolutionary search over architecture and loss weights

NSGA-II (ported from the RUL project), population 8, 5 generations, 40 genomes trained. Objectives: validation velocity error (8 cases) and parameter count, both minimised. Generation 0 includes the hand-tuned Phase 2 genome.
Proxy budget per genome: 400 Adam steps + 100 L-BFGS iterations. Total search cost: 40.1 min of training on NVIDIA GeForce RTX 2060 (clock capped at 1000 MHz).

## Search history

| Generation | Best error | Median error | Parameters (min–max) | Front size | Genomes trained |
|---|---|---|---|---|---|
| 0 | 0.05374 | 0.123 | 6595–48707 | 4 | 8 |
| 1 | 0.02412 | 0.06818 | 4483–48707 | 2 | 16 |
| 2 | 0.02412 | 0.06442 | 4483–48707 | 3 | 24 |
| 3 | 0.02412 | 0.04511 | 4483–6851 | 3 | 32 |
| 4 | 0.02412 | 0.04164 | 4483–39395 | 3 | 40 |

Hand-tuned Phase 2 genome at the proxy budget: error 0.05829, 18499 parameters.

## Final Pareto front

| Error | Parameters | Genome |
|---|---|---|
| 0.02412 | 6851 | 6×32 silu, lr 1.4e-03, N 192, w body/tan/kutta/far/data 5.41/6.61/15.4/3.16/63.9 |
| 0.03749 | 5795 | 5×32 tanh, lr 3.2e-04, N 384, w body/tan/kutta/far/data 42.6/0.15/1.02/0.282/9.07 |
| 0.148 | 4483 | 3×32 tanh, FF 16@0.96, lr 3.2e-04, N 192, w body/tan/kutta/far/data 29.4/0.15/1.02/1.24/9.07 |

## Champion vs hand-tuned, full Phase 2 budget

Champion: 6×32 silu, lr 1.4e-03, N 192, w body/tan/kutta/far/data 5.41/6.61/15.4/3.16/63.9.
Budget: 4000 Adam steps + up to 1000 L-BFGS iterations, seeds 0, 1, 2.

| Model | Val velocity error | Test velocity error | Test Cp RMS | Test cl error (circulation) | Training time [s] | Steps to 3e-2 | Steps to 1e-2 | Plateau step |
|---|---|---|---|---|---|---|---|---|
| Phase 2 (hand-tuned) | 0.0117 ± 0.003 | 0.0116 | 0.116 | 0.121 | 575 ± 30 | 3000–4500 | 5000 (1/3 seeds) | 5000 |
| Phase 3 champion | 0.00715 ± 0.0011 | 0.00776 ± 0.0017 | 0.0935 ± 0.016 | 0.0373 ± 0.005 | 653 ± 5.5 | 500–4250 | 4500–4750 | 5000 |

Test columns: mean over the seeds whose networks were saved (all three for the champion, seed 0 only for Phase 2).
