# PINNs for the steady 2D Euler equations (exploratory)

Code: [`src/pinncfd/pinn_euler_2d/`](../src/pinncfd/pinn_euler_2d/) ·
Lesson: [Lesson 5](lessons/05-toward-compressible-flow.md) ·
Notebook: [`05_euler.ipynb`](../notebooks/05_euler.ipynb) ·
Results: [`validation/results/euler_exploratory.md`](../validation/results/euler_exploratory.md)

## Equations

Steady, inviscid, compressible flow of an ideal gas ($\gamma = 1.4$) in conservation form:

```math
\frac{\partial \mathbf F(\mathbf U)}{\partial x} + \frac{\partial \mathbf G(\mathbf U)}{\partial y} = 0,
\qquad
\mathbf U = \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ E \end{pmatrix},
\quad
\mathbf F = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(E + p) \end{pmatrix},
\quad
\mathbf G = \begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(E + p) \end{pmatrix},
```

with $E = p/(\gamma - 1) + \rho(u^2 + v^2)/2$. The conservation form matters at a shock. Its weak
solutions obey the Rankine–Hugoniot jump conditions; a non-conservative form, written for the
primitive variables, admits shocks of the wrong strength and speed (LeVeque 2002, Ch. 12).

## The network

[`EulerPINN`](../src/pinncfd/pinn_euler_2d/euler.py) maps $(x, y)$ to the primitive variables
$(\rho, u, v, p)$, scaled by a reference state. Density and pressure pass through a softplus
shifted so that a zero network output gives the reference value; they stay positive, so the
energy and the sound speed are always defined. The residual is assembled from the fluxes of the
predicted state, differentiated by autograd. An optional artificial viscosity adds
$-\nu\nabla^2\mathbf U$ to the residual, smearing a shock over a width of order
$\nu/\Delta u$, where a smooth network can represent it.

## Exact solutions

**Ringleb flow** (Ringleb 1940; Chiocchia 1985) is smooth, isentropic and irrotational,
obtained by the hodograph method. With speed $q$, streamline label $k \ge q$ and sound speed
$c$, where $c^2 = 1 - \tfrac{\gamma-1}{2}q^2$:

```math
\rho = c^{2/(\gamma-1)}, \quad
J = \frac{1}{c} + \frac{1}{3c^3} + \frac{1}{5c^5} - \frac{1}{2}\ln\frac{1+c}{1-c},
\quad
x = \frac{1}{2\rho}\left(\frac{1}{q^2} - \frac{2}{k^2}\right) + \frac{J}{2},
\quad
y = \pm\frac{1}{k\rho q}\sqrt{1 - \frac{q^2}{k^2}} .
```

The domain $0.7 \le k \le 1.2$, $q \ge 0.5$ turns the flow around the end of a plate and holds
a smooth supersonic pocket where $q > \sqrt{2/(\gamma+1)} = 0.913$. Exact data is imposed on the
whole boundary. Tests check that the solution satisfies the Euler equations to $10^{-6}$ and
that streamlines are lines of constant $k$.

**Oblique shock on a wedge** (NACA Report 1135; Anderson, *Modern Compressible Flow*, Ch. 4).
A uniform stream at Mach $M$ meets a wedge of half-angle $\theta$. The weak root of

```math
\tan\theta = 2\cot\beta\,\frac{M^2\sin^2\beta - 1}{M^2(\gamma + \cos 2\beta) + 2}
```

gives the shock angle $\beta$; the normal Mach number $M\sin\beta$ sets the Rankine–Hugoniot
jumps. For $M = 2$, $\theta = 10°$: $\beta = 39.31°$, $p_2/p_1 = 1.7066$,
$\rho_2/\rho_1 = 1.4584$, $M_2 = 1.6405$, matching the NACA 1135 charts to their rounding.
Tests check those values and that the normal fluxes are continuous across the shock to
$10^{-12}$.

The PINN domain is $[0, 1]^2$ above the wedge surface. The freestream is imposed on the inflow
($x = 0$) and the top ($y = 1$, which the shock does not reach), slip on the wall, and nothing
on the supersonic outflow.

## Variants tried on the wedge

| Variant | Residual |
|---|---|
| Plain | $\lVert \partial_x\mathbf F + \partial_y\mathbf G \rVert^2$ |
| Artificial viscosity | $\lVert \partial_x\mathbf F + \partial_y\mathbf G - \nu\nabla^2\mathbf U \rVert^2$, $\nu = 0.01$ |
| Down-weighted at the shock | $\lambda\lVert \partial_x\mathbf F + \partial_y\mathbf G \rVert^2$, $\lambda = 1/(1 + \lvert\nabla\cdot\mathbf v\rvert)$ (detached) |

The third is a simplified form of the weighted-equations idea of Liu et al. (2024): where the
flow compresses sharply, the residual is huge and uninformative, so it is down-weighted and the
smooth regions are not sacrificed to the shock. Mao, Jagtap & Karniadakis (2020) give the
earlier picture of PINNs on high-speed flows.

## Results

5 × 64 tanh network, 8192 interior points, 6000 Adam steps then up to 2000 L-BFGS iterations, on
the RTX 2060 at 1000 MHz:

| Case | State error | Away from the shock | Shock angle (exact 39.31°) | Training time |
|---|---|---|---|---|
| Ringleb flow (2 seeds) | 0.14 % | — | — | 421 s |
| Wedge, plain | 5.3 % | 3.0 % | 37.5° | 448 s |
| Wedge, artificial viscosity | 4.6 % | 2.3 % | 38.4° | 1336 s |
| Wedge, down-weighted at the shock | 5.6 % | 2.8 % | 36.9° | 580 s |

The smooth case works as well as the potential-flow PINNs. On the wedge, the post-shock state
is within 2 % of Rankine–Hugoniot, but the shock is smeared over 0.13–0.17 of the domain
(10–90 % rise, normal to the shock), seven times the viscous width, and placed too shallow.
54–75 % of the squared residual sits in the tenth of the domain around the shock, and every
variant plateaus near 5 % error. Lesson 5 analyses why.

## References

- F. Ringleb, "Exakte Lösungen der Differentialgleichungen einer adiabatischen Gasströmung",
  *ZAMM* 20, 1940.
- G. Chiocchia, "Exact solutions to transonic and supersonic flows", AGARD AR-211, 1985.
- Ames Research Staff, "Equations, tables, and charts for compressible flow", NACA Report 1135,
  1953.
- J. D. Anderson, *Modern Compressible Flow*, 3rd ed., Ch. 4.
- R. J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*, Cambridge, 2002.
- Z. Mao, A. D. Jagtap & G. E. Karniadakis, "Physics-informed neural networks for high-speed
  flows", *Comput. Methods Appl. Mech. Eng.* 360, 2020.
- L. Liu, S. Liu, H. Xie, F. Xiong, T. Yu, M. Xiao, L. Liu & H. Yong, "Discontinuity computing
  using physics-informed neural networks", *J. Sci. Comput.* 98, 2024.
