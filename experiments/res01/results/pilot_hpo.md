# RES-01 pilot: tuning methods on case 2 at an equal budget

Joukowski section at 4°, gene space `JOUKOWSKI_SPACE`, 600 s of device time per method (training + evaluation), proxy budget 400 Adam + 100 L-BFGS per candidate, validation set only. Each method's first candidate is the hand-tuned genome. One search seed: a pilot, not a result.

| method | candidates | spent [s] | hand-tuned (proxy) val error | best val error | best genome |
|---|---|---|---|---|---|
| nsga | 8 | 879.3 | 0.37 | 0.208 | 6×32 sin, FF 32@0.79, compress features, lr 9.2e-04, N 4096, w body/tan/far 46.6/63.4/1.18 |
| tpe_asha | 14 | 619 | 0.37 | 0.186 | 5×96 silu, FF 32@0.94, compress features, lr 2.3e-03, N 2048, w body/tan/far 5.54/44.2/12 |
| random | 6 | 625.7 | 0.37 | 0.186 | 5×96 silu, FF 32@0.94, compress features, lr 2.3e-03, N 2048, w body/tan/far 5.54/44.2/12 |
