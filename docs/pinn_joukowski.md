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

## The network: three basis flows

Potential flow is linear, so the network does not learn the flow at one incidence. It learns
three *basis* flows around the section, each with the body a streamline:

```math
\psi_x = y + \psi'_x, \qquad
\psi_y = -x + \psi'_y, \qquad
\psi_c = \frac{L(x, y)}{2\pi} + \psi'_c,
```

with $\psi'_x$, $\psi'_y$ and $\psi'_c$ the three outputs of one network. The first two are
unit freestreams along $x$ and $y$ without circulation, and the third is unit circulation. The
flow at incidence $\alpha$ is

```math
\psi = \cos\alpha \, \psi_x + \sin\alpha \, \psi_y + \Gamma \psi_c ,
```

and $\Gamma$ is **not learned**. It is the value that makes the superposed flow meet the Kutta
condition, in closed form from the three basis flows' trailing-edge velocity jumps
$K_x$, $K_y$ and $K_c$:

```math
\Gamma = -\frac{(K_x\cos\alpha + K_y\sin\alpha) K_c}{K_c^2 + \epsilon^2}, \qquad \epsilon = 10^{-3},
```

exactly as a panel method fixes its circulation. $\epsilon$ only matters while $K_c$, about 2.7
when trained, is still near zero. The jumps are recomputed before every evaluation and enter
no loss, so no basis flow sees a Kutta loss.

$L$ is the mean of $\ln|z - s|$ over a uniform vortex sheet $s$ on the camber line from 0.1 to
0.9 chord (eight straight segments, integrated in closed form). This is the bound vortex sheet
of thin-airfoil theory at uniform strength. Far away it is $\ln r$, like a point vortex, but it
has no point vortex's speeds near the surface. Each basis flow also has its own body value,
three learnable scalars in all. One network per incidence, as before. The input features are
a gene:

| `features` | What the network sees | In the search space |
|---|---|---|
| `plain` | $(x - 0.5, y)$ in chords | yes |
| `compress` | the airfoil family's radially compressed coordinates | yes |
| `exact_map` | log-polar coordinates of this section's circle plane | **no**: reference only |

On a Joukowski section the inverse Joukowski map is the solution's own coordinate system, so
`exact_map` hands the network most of the answer. It measures that gift and is never offered to
a search. Fourier features can be added on top of any of them by the genome.

### Why superposition: four failures of a learned circulation

The first version learned $\Gamma$ as a scalar beside a point vortex at mid-chord,
$\psi = y\cos\alpha - x\sin\alpha + \Gamma \ln r_c / 2\pi + \psi'$. In every run $\Gamma$ stayed
near zero (0.014 against 0.550 at 4°), yet the exact-map network reached a velocity error of
0.02 with its lift 20 % low. Diagnosis at 4°, hand-tuned genome, exact-map features:

1. **$\psi'$ carried the circulation.** The flux of $\psi'$ through a ring at 0.75 chord was
   0.079, about 90 % of $\Gamma/2\pi$, and fell to zero by the far-field circle at 8 chords.
   The network built the circulation near the body and cancelled it in the sparsely sampled
   far field, with a non-harmonic patch. A bounded harmonic $\psi'$ has zero flux through
   every enclosing curve, so the far term now also penalises the mean of $r\partial\psi'/\partial r$
   on rings at 0.75, 1.5 and 3 chords (128 points each). This removed the leak, but $\Gamma$
   still did not grow (0.06).
2. **The point vortex is stiff.** At mid-chord it sits about 0.05 chord from the surface of
   this thin section and induces speeds near 1.75 there, which $\psi'$ must cancel. Even with
   $\Gamma$ frozen at its exact value the error stayed at 0.17. The vortex sheet brought that
   to 0.05.
3. **A Kutta loss can be cheated.** With the sheet and the rings, $\Gamma$ rose but stalled
   near 0.40, while the Kutta loss read about $10^{-7}$. The network met the condition with a
   local distortion at the trailing edge that the mean Laplace residual barely charges.
   Spreading the condition over 8 stations, or weighting the near-surface residual
   differently, did not stop it. Superposition removes the cheat: no basis flow is trained on
   the Kutta condition.
4. **The circulation settles last.** $\Gamma$ follows from small differences between the
   basis flows' jumps at the edge, which converge only in the L-BFGS stage. At 4° with exact-map
   features, $\Gamma$ was 0.79 after 500 L-BFGS iterations and 0.60 after 2000, hence the
   3000-iteration second stage of the pilot.

## Loss terms

Physics only, summed over the three basis flows: the Laplace residual of $\psi'$ (scaled by
$\min(1, (d/0.1)^2)$ next to the surface), $\psi - \psi_b$ on the body, the normal velocity on
the body, and the far field: $\psi'^2 + (r \partial\psi'/\partial r)^2$ on $r = 8$ chords plus
the squared ring fluxes above. There is no Kutta loss and no Kutta weight gene.

A test checks that the exact basis flows superpose to the exact flow. Another checks that they
zero every loss term up to stated residuals: the far-field truncation, and the polygonal
surface normals, whose error is largest in the vertical-freestream flow that runs fast round
the nose and the cusp. A third checks that they have no ring flux, while the full circulation
flow has $1/2\pi$.

**The Kutta condition on a cusp.** The airfoil family penalises the jump in downstream
tangential velocity just above and below the trailing edge. On a *cusped* edge that is
biased: a loaded cusp keeps a jump that shrinks only like $\sqrt{1 - x}$, so a flow with 10 %
less circulation scores lower than the exact one. Case 2 measures the jump $\Delta$ at 8
stations from $x = 0.85$ to $0.985$ and fits a straight line in $s = \sqrt{1 - x}$; the
intercept is the jump at the edge, $K$ above. For the exact flow it is $6 \times 10^{-4}$, and
for 90 % of its circulation 0.15. A test checks that the Kutta circulation of the exact basis
flows is the exact $\Gamma$ to 0.3 % at 0°, 4° and 8°.

**Phase 2's low lift slope (a hypothesis, not tested).** The airfoil family
([Lesson 2](lessons/02-airfoil-flow-fields.md)) has the same three ingredients that failed here: a learned $\Gamma$
beside a point vortex, a $\psi'$ checked against the far field only at 8 chords, and a Kutta
loss on the combined field. Its lift slope is about 20 % low, the same size as case 2's
before the fix. Mechanisms 1 and 3 plausibly explain it. This is not verified, and Phases 1–5
are unchanged.

## Metrics

Two fixed, disjoint evaluation sets: validation (what a search may see) and test (touched once
for reporting). Each has 2048 field points at least 0.005 chord off the surface and 400 body
points interleaved with the other set's. Reported: the relative $L_2$ velocity error (the
`error`), the surface $C_p$ RMS and maximum, the lift error from the circulation on a ring and
from integrating the pressure over the same panels as the exact $C_p$, and the error of the
Kutta circulation $\Gamma$.

## Pilot (one seed, hand-tuned genome)

[`experiments/res01/results/pilot_joukowski.md`](../experiments/res01/results/pilot_joukowski.md),
3000 Adam steps and 3000 L-BFGS iterations per run (17–31 minutes on the RTX 2060):

| Features | Test velocity error, 0° / 4° / 8° | $\Gamma$, 0° / 4° / 8° | Test lift error from the ring circulation, 0° / 4° / 8° |
|---|---|---|---|
| exact | – | 0.312 / 0.550 / 0.785 | – |
| exact_map (reference) | 0.0039 / 0.0059 / 0.0083 | 0.306 / 0.548 / 0.788 | 1.4 % / 0.7 % / 0.02 % |
| plain | 0.80 / 0.81 / 0.82 | −0.001 at every angle | 62 % / 110 % / 157 % |
| compress | – / 0.71 / – | −0.001 | – / 110 % / – |

**With the exact-map features the fix works.** The circulation is within 2 % at every angle
and the lift is right, where the learned $\Gamma$ stayed near zero and the lift was 11–44 %
off at 500 L-BFGS iterations. The velocity error fell from 0.021–0.024 to 0.004–0.008.

**With plain or compressed coordinates it still fails, and that is a finding, not a bug.** The
basis flows are singular at the cusp, where the conformal map has a branch point, and a network
on ordinary coordinates cannot represent them: the Laplace residual stalls near 0.09, the
trailing-edge jumps are noise, and the Kutta circulation is about zero. Longer training makes
it worse. In a diagnostic run on plain coordinates the velocity error rose from 0.31 after 250
Adam steps to 0.72 at the end, and the hand-tuned genome scores 0.37 at the tuning pilot's
short proxy budget but 0.81 at the full one. So case 2 asks a sharp question of a search: can
it find inputs on which these basis flows are learnable? In the
[tuning pilot](hpo_baseline.md), all three methods chose compressed coordinates with Fourier
features and got 0.19–0.21 at the proxy budget. Whether that survives the full budget is open.
One way to give plain coordinates a chance is to add the known trailing-edge singularity to the
known part of each basis flow; that is not built.
