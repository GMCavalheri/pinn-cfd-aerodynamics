# RES-01 pilot: case 2, hand-tuned genome by input features

One seed (0) per row; budget 3000 Adam steps + 3000 L-BFGS iterations. Γ is the circulation the Kutta condition gives the learned basis flows (unit chord and speed). `exact_map` is the reference that hands the network the solution's own coordinates; it is never offered to a search. A pilot, not a result: it calibrates the pre-registration.

| features | α | val velocity error | test velocity error | Γ | exact Γ | test Cp RMS (5–95 % chord) | test cl error (circulation) | test cl error (pressure) | training time [s] |
|---|---|---|---|---|---|---|---|---|---|
| plain | 0° | 0.801 | 0.801 | -0.000983 | 0.312 | 1.27 | 0.622 | 0.624 | 1.02e+03 |
| plain | 4° | 0.808 | 0.809 | -0.00095 | 0.55 | 1.42 | 1.1 | 1.1 | 1.02e+03 |
| plain | 8° | 0.822 | 0.821 | -0.000911 | 0.785 | 1.64 | 1.57 | 1.57 | 1.07e+03 |
| compress | 4° | 0.708 | 0.709 | -0.00115 | 0.55 | 1.37 | 1.1 | 1.09 | 1.47e+03 |
| exact_map | 0° | 0.00415 | 0.00387 | 0.306 | 0.312 | 0.0244 | 0.0137 | 0.00701 | 1.73e+03 |
| exact_map | 4° | 0.00631 | 0.00585 | 0.548 | 0.55 | 0.0425 | 0.00676 | 0.00221 | 1.87e+03 |
| exact_map | 8° | 0.00891 | 0.00825 | 0.788 | 0.785 | 0.0611 | 0.000182 | 0.0113 | 1.75e+03 |
