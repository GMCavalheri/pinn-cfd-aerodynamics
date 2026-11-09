# RES-01 pilot: tuning methods on case 2 at an equal budget

Joukowski section at 4°, gene space `JOUKOWSKI_SPACE`, 600 s of device time per method (training + evaluation), proxy budget 400 Adam + 100 L-BFGS per candidate, validation set only. Each method's first candidate is the hand-tuned genome. One search seed: a pilot, not a result.

| method | candidates | spent [s] | hand-tuned (proxy) val error | best val error | best genome |
|---|---|---|---|---|---|
| nsga | 24 | 718.4 | 0.588 | 0.133 | 6×48 silu, FF 16@0.88, compress features, lr 1.0e-03, N 2048, w body/tan/kutta/far 10/6.07/10.4/5.41 |
| tpe_asha | 65 | 601.6 | 0.588 | 0.103 | 5×64 silu, FF 32@0.96, compress features, lr 1.1e-03, N 1024, w body/tan/kutta/far 0.45/5.48/0.776/2.51 |
| random | 21 | 605.6 | 0.588 | 0.142 | 6×48 silu, FF 16@1.66, compress features, lr 1.5e-03, N 2048, w body/tan/kutta/far 0.2/57.3/13.9/99.2 |
