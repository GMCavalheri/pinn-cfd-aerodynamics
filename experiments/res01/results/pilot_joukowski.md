# RES-01 pilot: case 2, hand-tuned genome by input features

One seed (0) per row; budget 3000 Adam steps + 500 L-BFGS iterations. `exact_map` is the reference that hands the network the solution's own coordinates; it is never offered to a search. A pilot, not a result: it calibrates the pre-registration.

| features | α | val velocity error | test velocity error | test Cp RMS (5–95 % chord) | test cl error (circulation) | test cl error (pressure) | training time [s] |
|---|---|---|---|---|---|---|---|
| plain | 0° | 0.784 | 0.784 | 1.26 | 0.687 | 0.621 | 122 |
| plain | 4° | 0.795 | 0.796 | 1.42 | 1.16 | 1.1 | 122 |
| plain | 8° | 0.806 | 0.805 | 1.64 | 1.66 | 1.57 | 122 |
| compress | 4° | 0.704 | 0.704 | 1.37 | 0.763 | 1.09 | 192 |
| exact_map | 0° | 0.0205 | 0.0208 | 0.0664 | 0.107 | 0.0585 | 271 |
| exact_map | 4° | 0.0224 | 0.0223 | 0.0641 | 0.228 | 0.1 | 286 |
| exact_map | 8° | 0.0245 | 0.0243 | 0.0778 | 0.439 | 0.112 | 308 |
