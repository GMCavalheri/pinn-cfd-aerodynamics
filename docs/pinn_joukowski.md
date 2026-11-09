# RES-01 case 2: one Joukowski section, exact flow and a PINN

Code: [`src/pinncfd/pinn_joukowski/`](../src/pinncfd/pinn_joukowski/) ·
Pilot: [`experiments/res01/results/pilot_joukowski.md`](../experiments/res01/results/pilot_joukowski.md) ·
Tests: [`test_joukowski_exact.py`](../tests/test_joukowski_exact.py),
[`test_joukowski_pinn.py`](../tests/test_joukowski_pinn.py)

This module serves the RES-01 paper (neuroevolutionary PINNs for aerodynamic flow), not the
course. Case 2 asks how well a PINN's configuration can be chosen when the network sees the
plane through ordinary coordinates, on a section whose flow is known exactly everywhere.

## The exact flow everywhere

The section is AERO-01's validation airfoil, `Joukowski(0.1, 0.1)`: a cusped, cambered
Joukowski airfoil, normalised to chord 1 with the leading edge at $x = 0$. The map

```math
z = \zeta + \frac{b^2}{\zeta}
```

sends the circle $|\zeta - \zeta_0| = a$, which passes through $\zeta = b$, onto the airfoil.
With $\zeta' = \zeta - \zeta_0$, the flow at incidence $\alpha$ with clockwise circulation
$\Gamma$ has the complex potential (Anderson, *Fundamentals of Aerodynamics*, Sec. 4.14; Katz &
Plotkin, Sec. 6.6)

```math
F(\zeta') = \zeta' e^{-i\alpha} + \frac{a^2 e^{i\alpha}}{\zeta'} + \frac{i\Gamma}{2\pi}\ln\zeta',
\qquad
u - i v = \frac{dF/d\zeta}{dz/d\zeta},
\qquad
\psi = \operatorname{Im} F .
```

The Kutta condition puts a stagnation point at $\zeta = b$, where $dz/d\zeta = 0$, and gives
$\Gamma = 4\pi a \sin(\alpha + \beta)$. The stream function is single-valued, because the
imaginary part of $i c \ln\zeta'$ is $c \ln|\zeta'|$. AERO-01 provides the surface pressure;
`JoukowskiFlow` extends it to the whole field by inverting the map (the root outside the
circle). The tests check that the body is a streamline, that the surface pressure matches
AERO-01's to $10^{-10}$, that $\psi$ is harmonic with the velocity as its curl, that the
circulation on a ring gives the Kutta-Joukowski lift, that the far field is the freestream and
that the speed stays finite at the cusp.

## The network

```math
\psi(x, y) = y\cos\alpha - x\sin\alpha + \frac{\Gamma}{2\pi}\ln r_c + \psi'(x, y),
```

with $r_c$ the distance to the camber line at mid-chord, and $\Gamma$ and the body value
$\psi_b$ learnable scalars. One network per incidence. The input features are a gene:

| `features` | What the network sees | In the search space |
|---|---|---|
| `plain` | $(x - 0.5, y)$ in chords | yes |
| `compress` | the airfoil family's radially compressed coordinates | yes |
| `exact_map` | log-polar coordinates of this section's circle plane | **no**: reference only |

On a Joukowski section the inverse Joukowski map is the solution's own coordinate system, so
`exact_map` hands the network most of the answer. It measures that gift and is never offered to
a search. Fourier features can be added on top of any of them by the genome.

## Loss terms

As in the airfoil family (Lesson 2), physics only: the Laplace residual of $\psi'$ (scaled by
$\min(1, (d/0.1)^2)$ next to the surface), $\psi - \psi_b$ on the body, the normal velocity on
the body, the Kutta condition, and $\psi'^2 + (r \partial\psi'/\partial r)^2$ on $r = 8$ chords.

**The Kutta condition had to change.** The family penalises the jump in downstream tangential
velocity just above and below the trailing edge, at $x = 0.97$ and $0.985$. On a *cusped* edge
that is biased: a loaded cusp keeps a velocity jump that shrinks only like $\sqrt{1 - x}$, so
the exact flow leaves the loss at $1.3 \times 10^{-2}$ at 4°, while a flow with 10 % less
circulation scores $1.4 \times 10^{-3}$. The loss is minimised by too little lift. Case 2
measures the jump $\Delta$ at the same two stations and extrapolates it linearly in
$s = \sqrt{1 - x}$ to the edge:

```math
\Delta_{\mathrm{TE}} = \frac{s_2 \Delta_1 - s_1 \Delta_2}{s_2 - s_1},
\qquad
L_{\mathrm{kutta}} = \Delta_{\mathrm{TE}}^2 .
```

The exact flow leaves this at $3 \times 10^{-7}$, and a flow with 90 % of its circulation at
$5 \times 10^{-2}$. A test checks both numbers, and another checks that the exact flow zeroes
every loss term up to stated residuals (the far-field truncation and the polygonal surface
normals).

Whether the family's condition causes Phase 2's low lift slope on the NACA sections (finite
trailing-edge angle, not a cusp) is an open question: the bias found here suggests it.

## Metrics

Two fixed, disjoint evaluation sets: validation (what a search may see) and test (touched once
for reporting). Each has 2048 field points at least 0.005 chord off the surface and 400 body
points interleaved with the other set's. Reported: the relative $L_2$ velocity error (the
`error`), the surface $C_p$ RMS and maximum, and the lift error from the circulation on a ring
and from integrating the pressure over the same panels as the exact $C_p$.

## Pilot (one seed, hand-tuned genome)

[`experiments/res01/results/pilot_joukowski.md`](../experiments/res01/results/pilot_joukowski.md),
3000 Adam steps and 500 L-BFGS iterations per run:

| Features | Test velocity error, 0° / 4° / 8° | Test lift error from the ring circulation, 4° |
|---|---|---|
| plain | 0.78 / 0.80 / 0.81 | 1.16 |
| compress | – / 0.70 / – | 0.76 |
| exact_map (reference) | 0.021 / 0.022 / 0.024 | 0.23 |

Plain coordinates fail here as they did in Phase 2, so case 2 does test whether a search can
find inputs that work. Compressed coordinates alone do not rescue it, but every search in the
[tuning pilot](hpo_baseline.md) moved to compressed coordinates with Fourier features and got
0.10–0.14 at a fifth of this budget.

**Open issue: the learned circulation.** In every run the scalar $\Gamma$ stays near zero
(at most 0.03, against about 0.55 exact at 4°). The exact-map network carries its circulation in
$\psi'$ instead, through its $\ln|\zeta'|$ input, which the far-field term penalises; its lift
from the ring circulation is about 20 % low even though the Kutta loss is satisfied to
$10^{-9}$. That is the size of Phase 2's low lift slope, so the far-field term and the
circulation parameterisation, not only the biased Kutta condition, are suspects. This must be
resolved before the pre-registration.
