# PINN for potential flow past a cylinder

Code: [`src/pinncfd/pinn_laplace_2d/`](../src/pinncfd/pinn_laplace_2d/) ·
Lesson: [Lesson 1](lessons/01-pinns-for-potential-flow.md) ·
Notebook: [`01_laplace_cylinder.ipynb`](../notebooks/01_laplace_cylinder.ipynb) ·
Results: [`validation/results/laplace_cylinder.md`](../validation/results/laplace_cylinder.md)

## The boundary-value problem

Incompressible, irrotational 2D flow has a stream function $\psi$ with $u = \partial\psi/\partial y$,
$v = -\partial\psi/\partial x$, and $\nabla^2\psi = 0$. For a unit cylinder in a unit freestream
with clockwise circulation $\Gamma$, the exact solution is uniform flow plus a doublet plus a
vortex (Anderson, *Fundamentals of Aerodynamics*, Sec. 3.13–3.15):

```math
\psi = \left(r - \frac{1}{r}\right)\sin\theta + \frac{\Gamma}{2\pi}\ln r,
\qquad
C_p(\theta) = 1 - \left(2\sin\theta + \frac{\Gamma}{2\pi}\right)^2,
\qquad
c_l = \Gamma .
```

(The lift coefficient is per unit diameter: $L' = \rho V \Gamma$ with diameter 2.)

The PINN solves it on the annulus $1 \le r \le R$, $R = 5$, with the exact $\psi$ as Dirichlet
data on $r = R$. This is a verification problem: the question is how well the network solves a
well-posed boundary-value problem, not how well a truncated far field stands in for infinity.

## Why the stream function

The velocity potential $\phi$ of a lifting flow jumps by $\Gamma$ around the body, so a
network, which is continuous, cannot represent it without a branch cut. $\psi$ is single-valued
with or without circulation, and the body condition becomes Dirichlet ($\psi = $ const on the
surface) instead of Neumann ($\partial\phi/\partial n = 0$). The project uses $\psi$ throughout;
$\phi$ appears only for the non-lifting cylinder, as a comparison.

## Four formulations

| Formulation | Output | Loss terms |
|---|---|---|
| $\psi$, soft constraints | $\psi = N$ | Laplace residual, $\psi = 0$ on $r = 1$, $\psi = \psi_R$ on $r = R$ |
| $\psi$, hard constraints | $\psi = \psi_b + dN$ | Laplace residual only |
| $\psi$, hard, $\Gamma = 2\pi$ | as above | Laplace residual only |
| $\phi$, soft (Neumann body) | $\phi = N$ | Laplace residual, $\partial\phi/\partial n = 0$ on $r = 1$, $\phi = \phi_R$ on $r = R$ |

The hard-constraint ansatz (Lagaris, Likas & Fotiadis 1998; Sukumar & Srivastava 2022) is

```math
\psi = \psi_R(\theta)\frac{r - 1}{R - 1} + d(r) N(x, y),
\qquad
d = \frac{(r - 1)(R - r)}{\left((R - 1)/2\right)^2},
```

which meets both boundary conditions for *any* network output $N$, so only the residual is left
to minimise.

## Results (mean over three seeds)

| Formulation | Velocity error | Steps to 1e-2 | Steps to 1e-3 | Training time |
|---|---|---|---|---|
| $\psi$, soft | 1.2e-3 | 5600–5700 | never | 168 s |
| $\psi$, hard | 7.7e-5 | 400–500 | 1300–1500 | 183 s |
| $\psi$, hard, $\Gamma = 2\pi$ | 6.7e-5 | 300–400 | 1000–1400 | 194 s |
| $\phi$, soft | 1.9e-4 | 4600–5100 | 5700–5800 | 177 s |

4 × 50 tanh network, 8192 interior points, 5000 Adam steps then up to 4000 L-BFGS iterations;
RTX 2060 at 1000 MHz. Hard constraints are 15 times more accurate and reach 1 % error more than
ten times sooner. With soft constraints the boundary terms and the residual pull against each
other: the residual is driven down while the boundary misfit stalls (Wang, Teng & Perdikaris
2021). A 64–512-panel AERO-01 solution of the same flow reaches 6.9e-3 to 9.3e-4 velocity
error, converging at first order in the panel count.

## References

- J. D. Anderson, *Fundamentals of Aerodynamics*, 6th ed., Sec. 3.13–3.15.
- I. E. Lagaris, A. Likas & D. I. Fotiadis, "Artificial neural networks for solving ordinary
  and partial differential equations", *IEEE Trans. Neural Netw.* 9, 1998.
- M. Raissi, P. Perdikaris & G. E. Karniadakis, "Physics-informed neural networks", *J. Comput.
  Phys.* 378, 2019.
- N. Sukumar & A. Srivastava, "Exact imposition of boundary conditions with distance functions
  in physics-informed deep neural networks", *Comput. Methods Appl. Mech. Eng.* 389, 2022.
- S. Wang, Y. Teng & P. Perdikaris, "Understanding and mitigating gradient flow pathologies in
  physics-informed neural networks", *SIAM J. Sci. Comput.* 43, 2021.
