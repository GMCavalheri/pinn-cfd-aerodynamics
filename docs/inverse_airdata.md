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

The case-2 basis flows, with the incidence and speed fitted to the readings:

```math
\psi = U\left[\cos\hat\alpha \, \psi_x + \sin\hat\alpha \, \psi_y + \Gamma(\hat\alpha) \psi_c\right],
```

with $\Gamma(\hat\alpha)$ the Kutta circulation of the basis flows at $\hat\alpha$, as in
[case 2](pinn_joukowski.md#the-network-three-basis-flows). The basis flows do not depend on
$\alpha$ or $U$, so the readings carry no information about them. The network therefore trains
on case 2's physics alone. At every evaluation, $(\hat\alpha, U)$ are fitted by least squares
to the readings on the current basis flows, from the oracle's start (0° and $U = 1$), with the
residuals divided by the largest reading. It is the oracle's problem with the learned flow in
place of the exact one, so the gap to the oracle measures what the learned flow costs.
Reported: the incidence error in degrees (the `error`), the relative speed error, the RMS tap
residual, and the field velocity error against the exact flow at the true condition.

**Three versions failed first.** All were run at 4° with the reference features:

1. **The tap misfit as a loss term that also trained the network.** The network bent the
   basis flows to fit the readings at the starting incidence, and $\hat\alpha$ barely moved
   (4.2–4.5° off).
2. **The same with the basis flows detached in that term.** $\hat\alpha$ and $U$ chased
   basis flows that were far from converged, drifting to −20° in a short run. Refreshing the
   Kutta jumps inside the loss also changed the objective between L-BFGS line-search steps.
   L-BFGS then stopped after a few iterations, as it had in version 1 (60 of 3000), so the
   basis flows never converged.
3. **The least-squares fit warm-started from the previous fit.** An early fit to rough basis
   flows (55°, $U = 0.06$) stuck.

## Pilot (one seed, 8 taps, 4°)

[`experiments/res01/results/pilot_airdata.md`](../experiments/res01/results/pilot_airdata.md),
3000 Adam steps and 3000 L-BFGS iterations (18–30 minutes per run):

| Features | Noise | Incidence error | Speed error | $\Gamma / U$ (exact 0.550) | Field velocity error |
|---|---|---|---|---|---|
| exact_map (reference) | 0 % | 0.073° | 0.47 % | 0.544 | 0.009 |
| exact_map (reference) | 2 % | 0.060° | 0.04 % | 0.544 | 0.006 |
| plain | 0 % and 2 % | 30° | 100 % (fit collapses to $U \to 0$) | −0.001 | 1.0 |
| oracle, draw 0 | 2 % | 0.013° | 0.49 % | – | – |

With the reference features the PINN recovers the incidence to 0.06–0.07°, better than before
the case-2 fix (0.16–0.32°). That is about five times the oracle's error at 2 % noise, and the
same size without noise. So the floor is the learned flow, not the noise: the RMS tap residual
is 0.019 even with clean readings. One noisy draw and one seed cannot separate the two
noise levels.

On plain coordinates the basis flows are wrong (case 2's failure), and no incidence and speed
fit the readings. The least-squares fit collapses to $U \to 0$, which turns every predicted
reading to zero, leaving a residual of 0.67. Before the fix, plain coordinates showed a 1.2–1.4°
incidence error, but only because $\hat\alpha$ stayed near its start while the speed was 270 %
off. The estimate never carried information, so the collapse is the more honest failure.
