# PINN for a family of airfoil flow fields

Code: [`src/pinncfd/pinn_airfoil_flowfield/`](../src/pinncfd/pinn_airfoil_flowfield/) ·
Lesson: [Lesson 2](lessons/02-airfoil-flow-fields.md) ·
Notebook: [`02_airfoil_family.ipynb`](../notebooks/02_airfoil_family.ipynb) ·
Results: [`validation/results/airfoil_flowfield.md`](../validation/results/airfoil_flowfield.md)

## A parametric solver

One network gives the stream function of any NACA 4-digit section in a box of camber $m$,
camber position $p$, thickness $t$ and incidence $\alpha$:

| $m$ | $p$ | $t$ | $\alpha$ |
|---|---|---|---|
| 0–0.06 | 0.3–0.5 | 0.08–0.18 | −4° to 8° |

Splits ([`family.py`](../src/pinncfd/pinn_airfoil_flowfield/family.py)): 40 training cases with
panel data (Latin hypercube, seed 0); 8 validation cases (seed 1); a test set of NACA 0012, 1408,
2412, 3315, 4415 and 5518 at −3°, 1°, 4° and 7°, never seen before the final evaluation; and four
sections just outside the box. The physics terms draw fresh shapes from the whole box at every
resample, so they need no solved cases.

## The model

With $q = (m, p, t, \alpha)$:

```math
\psi(x, y; q) = y\cos\alpha - x\sin\alpha + \frac{\Gamma(q)}{2\pi}\ln\frac{|w|}{R} + \psi'(x, y; q).
```

- $w(x, y; q)$ is the inverse Joukowski map of a circular-arc slit inside the section, from
  $x = 0.55t^2$ inside the nose to the trailing edge, with height $0.9m$. It maps the exterior
  of the slit onto the exterior of a circle of radius $R$. $\ln|w|$ is harmonic in the fluid and behaves like $\ln r$
  far away, so the third term is a vortex of clockwise circulation $\Gamma$ "inside" the section.
- $\Gamma$ and the body value $\psi_b$ come from a small head network of $q$ alone.
- $\psi'$ is the main network. It sees the log-polar coordinates of the circle plane,
  $(\ln(|w|/R),\ w/|w|)$, plus $q$, not $(x, y)$. In those coordinates the section is a
  near-circle and the flow is smooth. The sharp leading- and trailing-edge behaviour is carried
  by the map's derivative instead of by the network.

The lift is always measured from the field: the circulation on a circle of radius 1.5 around the
section ($c_l = 2\Gamma$), or the integral of $C_p = 1 - |\mathbf V|^2$ over the surface. $\Gamma$
in the formula above is a convenient split, not a measurement.

## Loss terms

| Term | What it enforces | Why it has this form |
|---|---|---|
| `pde` | $\nabla^2\psi' = 0$ at interior points, scaled by $\min(1, (d/0.1)^2)$ | near the leading edge $\nabla^2\psi$ is O(100), so a small relative error would swamp the loss |
| `body` | $\psi = \psi_b$ on the surface | the body is a streamline |
| `tangency` | $\mathbf V\cdot\mathbf n = 0$ on the surface | a leak of flux $Q$ through a stretch $\ell$ costs $O(Q^2)$ in `body` but $O(Q^2/\ell^2)$ here |
| `kutta` | equal *downstream* tangential velocities just ahead of the trailing edge | equal speeds also hold for the circulation-free flow |
| `far` | $\psi'^2 + (r \partial\psi'/\partial r)^2$ on $r = 8$ | forbids both a hidden log term and a large doublet |
| `data` | velocity misfit to AERO-01 (400 panels) at field points of training cases | supplies the lift that physics alone under-predicts |

Each form fixes a failure seen in a first version. Lesson 2 documents nine of them.

## Results (test set)

| Model | Velocity error | Cp RMS | cl error (circulation) | Training time |
|---|---|---|---|---|
| Physics + panel data, 5×64 tanh (Phase 2) | 1.16 % | 0.116 | 0.121 | 575 s |
| Physics only | 2.1 % | 0.120 | 0.181 | 464 s |
| Panel data only | 0.88 % | 0.325 | 0.035 | 89 s |
| Plain $(x, y)$ inputs | 2.5 % | 0.301 | 0.331 | 361 s |
| Phase 3 champion, 6×32 SiLU, data weight 64 (3 seeds) | 0.78 % | 0.094 | 0.037 | 653 s |

Physics improves the surface pressure, where there is no data; data fixes the lift. The
conformal-map inputs mattered more than any loss weight: without them the physics-only network
reached 60–90 % error in the early attempts.

## References

- L. M. Milne-Thomson, *Theoretical Aerodynamics*, Ch. 6–7.
- I. H. Abbott & A. E. von Doenhoff, *Theory of Wing Sections*, Sec. 6.4.
- J. Katz & A. Plotkin, *Low-Speed Aerodynamics*, Sec. 6.6 and Ch. 11.
- M. D. McKay, R. J. Beckman & W. J. Conover, *Technometrics* 21, 1979.
