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

The case-2 basis flows with the incidence and speed learnable:

```math
\psi = U\left[\cos\hat\alpha \, \psi_x + \sin\hat\alpha \, \psi_y + \Gamma(\hat\alpha) \psi_c\right],
```

with $\hat\alpha$, $\log U$ and the three body values trained together with the network,
starting from $\hat\alpha = 0$ and $U = 1$. $\Gamma(\hat\alpha)$ is the Kutta circulation of
the current basis flows at the current $\hat\alpha$, recomputed every step, as in
[case 2](pinn_joukowski.md#the-network-three-basis-flows). The loss is case 2's physics plus
the tap misfit, divided by the largest reading (about the stagnation pressure, so the term is
O(1) without using the true speed). Reported: the incidence error in degrees (the `error`), the
relative speed error, and the field velocity error against the exact flow at the true
condition.

## Pilot (one seed, 8 taps, 4°)

[`experiments/res01/results/pilot_airdata.md`](../experiments/res01/results/pilot_airdata.md),
3000 Adam steps and 3000 L-BFGS iterations:

| Features | Noise | Incidence error | Speed error | $\Gamma / U$ (exact 0.550) | Before the basis flows (incidence, speed) |
|---|---|---|---|---|---|
| plain | 0 % | 1.37° | 271 % | −0.009 | 1.23°, 13.8 % |
| plain | 2 % | 1.22° | 318 % | −0.007 | 2.41°, 15.3 % |
| exact_map (reference) | 0 % | 4.23° | 18.3 % | 0.092 | 0.16°, 0.13 % |
| exact_map (reference) | 2 % | 4.49° | 16.2 % | 0.100 | 0.32°, 1.1 % |
| oracle, draw 0 | 2 % | 0.013° | 0.49 % | – | – |

**A regression.** The case-2 fix made the inverse problem worse. With the reference features
$\hat\alpha$ barely moves from its 0° start (−0.2° at the end), the circulation is a fifth of
the exact one, and the speed is 18 % high. The tap misfit trains the network as well as
$\hat\alpha$ and $U$, so the network bends the basis flows to fit the readings at the wrong
incidence, and that also spoils their trailing-edge jumps and so $\Gamma$. With superposition the
basis flows do not depend on $\alpha$ or $U$ at all, so the readings carry no information about
them. A fix that follows from this is to let the tap term train only $\hat\alpha$ and $U$
(detach the basis flows in it), which makes the PINN a learned-flow version of the oracle. That
is a change to what the inverse PINN is, so it waits for a decision, and these numbers stand as
the pilot.
