# RES-01 pilot: inverse air data

## Oracle noise floor

Least squares on the exact pressure model; 20 validation noise draws at each of 0°, 4° and 8°, U = 1.15. Errors are absolute for α, relative for U.

| taps | noise | α error, mean [°] | α error, 95th pct [°] | U error, mean |
|---|---|---|---|---|
| 4 | 0 % | 2.11e-15 | 3.18e-15 | 6.44e-17 |
| 4 | 1 % | 0.0472 | 0.13 | 0.0041 |
| 4 | 2 % | 0.0981 | 0.289 | 0.00809 |
| 4 | 5 % | 0.201 | 0.614 | 0.0167 |
| 6 | 0 % | 1.27e-15 | 2.39e-15 | 6.44e-17 |
| 6 | 1 % | 0.0199 | 0.048 | 0.00239 |
| 6 | 2 % | 0.0367 | 0.0844 | 0.00388 |
| 6 | 5 % | 0.108 | 0.236 | 0.0104 |
| 8 | 0 % | 8.73e-16 | 1.59e-15 | 0 |
| 8 | 1 % | 0.0143 | 0.0362 | 0.00143 |
| 8 | 2 % | 0.0322 | 0.0654 | 0.00436 |
| 8 | 5 % | 0.0972 | 0.196 | 0.0113 |
| 10 | 0 % | 5.66e-16 | 1.59e-15 | 6.44e-17 |
| 10 | 1 % | 0.0138 | 0.0342 | 0.00176 |
| 10 | 2 % | 0.0284 | 0.0634 | 0.00417 |
| 10 | 5 % | 0.0872 | 0.228 | 0.00892 |

## PINN (one seed, validation draw 0)

8 taps, true α = 4° and U = 1.15, started from 0° and U = 1. Budget 3000 Adam steps + 500 L-BFGS iterations. `exact_map` is the reference features (never offered to a search).

| features | noise | α error [°] | U error | oracle α error [°] | oracle U error | field velocity error | training time [s] |
|---|---|---|---|---|---|---|---|
| plain | 0 % | 1.23 | 0.138 | 7.95e-16 | 0 | 0.181 | 179 |
| plain | 2 % | 2.41 | 0.153 | 0.0134 | 0.00486 | 0.195 | 174 |
| exact_map | 0 % | 0.164 | 0.00127 | 7.95e-16 | 0 | 0.0196 | 334 |
| exact_map | 2 % | 0.316 | 0.0113 | 0.0134 | 0.00486 | 0.0161 | 344 |
