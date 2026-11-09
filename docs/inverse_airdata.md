# RES-01 inverse problem: air data from surface pressure taps

Code: [`src/pinncfd/inverse_airdata/`](../src/pinncfd/inverse_airdata/) ·
Pilot: [`experiments/res01/results/pilot_airdata.md`](../experiments/res01/results/pilot_airdata.md) ·
Tests: [`test_inverse_airdata.py`](../tests/test_inverse_airdata.py)

A flush air data system (FADS) estimates the flight condition from pressures at ports flush
with the skin around the nose, instead of from probes. Here the section of case 2 plays the
aircraft: recover the incidence $\alpha$ and the freestream speed $U$ from a few noisy surface
pressure readings, with the density and the freestream static pressure known. It is the seed of
later RES work on telling sensor faults from attacks: a wrong air-data estimate is a safety
problem whatever its cause.

## Taps and readings

Layouts of 4, 6, 8 and 10 taps, symmetric about the leading edge in the circle-plane angle and
spanning ±60° of it, so they cluster on the curved nose. With $\rho = 1$, each reading is

```math
\Delta p = p - p_\infty = \tfrac12\left(U^2 - |V|^2\right) = \tfrac12 U^2 C_p(\alpha) + \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}\!\left(0, \left(\sigma \tfrac12 U^2\right)^2\right),
```

with $\sigma$ = 0, 1, 2 or 5 % of the dynamic pressure. Noise draws come from fixed seeds,
disjoint between the validation draws a search may use and the test draws used for reporting.

## The oracle

`oracle.fit` solves the least-squares problem for $(\alpha, U)$ with the *exact* pressure
model, as a calibrated FADS would. It is exact without noise (a test checks $10^{-10}$ on every
layout) and sets the noise floor: the PINN has to learn the flow *and* the air data from the
same readings, the oracle only the air data.

## The PINN

The case-2 network with the incidence and speed learnable:

```math
\psi = U\left[y\cos\hat\alpha - x\sin\hat\alpha + \frac{\Gamma}{2\pi}\ln r_c + \psi'(x, y)\right],
```

with $\hat\alpha$, $\log U$, $\Gamma$ and $\psi_b$ trained together with $\psi'$, starting from
$\hat\alpha = 0$ and $U = 1$. The loss is case 2's physics plus the tap misfit, divided by the
largest reading (about the stagnation pressure, so the term is O(1) without using the true
speed). Reported: the incidence error in degrees (the `error`), the relative speed error, and
the field velocity error against the exact flow at the true condition.

## Pilot (one seed, 8 taps, 4°)

[`experiments/res01/results/pilot_airdata.md`](../experiments/res01/results/pilot_airdata.md):

| Features | Noise | Incidence error | Speed error |
|---|---|---|---|
| plain | 0 % | 1.23° | 13.8 % |
| plain | 2 % | 2.41° | 15.3 % |
| exact_map (reference) | 0 % | 0.16° | 0.13 % |
| exact_map (reference) | 2 % | 0.32° | 1.1 % |
| oracle, mean of 60 draws | 2 % | 0.032° | 0.44 % |

Even with the reference features the PINN is about ten times the oracle's floor in incidence.
It shares case 2's circulation problem (see [case 2](pinn_joukowski.md#pilot-one-seed-hand-tuned-genome)),
and the incidence is read mostly from the circulation, so it is not yet a fair test of the
method.
