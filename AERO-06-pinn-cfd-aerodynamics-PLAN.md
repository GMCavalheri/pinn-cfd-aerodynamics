# PLAN.md — AERO-06: Physics-Informed Neural Networks for Aerodynamics (PINN-CFD)

## Overview
A differentiator project applying Physics-Informed Neural Networks (PINNs) to aerodynamic flow problems — solving and accelerating potential flow and simplified compressible flow problems by embedding governing PDEs directly into the neural network's loss function. This is a direct extension of the graduate monograph ("Aplicando Neuroevolução em Physics Informed Neural Networks") into the aerospace engineering domain.

## Objective
Build PINN-based solvers for classical aerodynamic PDEs (starting with the potential flow / Laplace equation, extending toward the 2D steady Euler equations), and compare their accuracy and computational cost against the classical panel-method/VLM solvers built in the Aerodynamics From Scratch project.

## Why This Project
This is the single highest-differentiation project in the portfolio. Very few candidates — including many aerospace engineering PhDs — combine hands-on aerodynamics implementation with physics-informed machine learning applied to that same domain. It directly demonstrates that your thesis work (neuroevolution + PINNs) isn't a one-off academic exercise but a transferable research capability you can apply to a new physical domain on demand. It's also a legitimate, active research area in the aerospace industry (surrogate models for CFD, design optimization loops).

## Knowledge Covered
1. **PINN Fundamentals for Fluid Flow**: implement a PINN that solves the 2D Laplace equation (potential flow around simple shapes) by embedding the PDE residual in the loss function, compare against the analytical/panel-method solution from Aerodynamics From Scratch.
2. **Airfoil Flow Field Prediction**: train a PINN to predict the full flow field (velocity potential, pressure) around a parameterized family of airfoil shapes, using boundary conditions from the geometry instead of only a single fixed shape.
3. **Extension Toward Compressible/Euler Flow**: extend the PINN formulation toward the steady 2D Euler equations for transonic-adjacent regimes, acknowledging the added difficulty of shock capturing (this phase is exploratory/research-grade and should be scoped as "best effort," not a hard deliverable).
4. **Neuroevolution for Architecture Search**: reuse the neuroevolution methodology from the thesis to optimize the PINN's architecture and the relative weighting between data loss and physics-residual loss — the same core contribution as the thesis, now applied to a new physical system.
5. **Surrogate Model Speed Comparison**: benchmark PINN inference time against the classical panel-method solver for repeated evaluations (e.g., inside a design optimization loop), demonstrating the practical surrogate-model value proposition of this approach.

## Prerequisites / Background
- Completion (or substantial progress) on Aerodynamics From Scratch, to have a classical baseline to compare against
- The PINN/neuroevolution methodology from the graduate monograph and the RUL Prediction (PINN + Neuroevolution) reliability project — this project reuses that same technical machinery in a new domain

## Data / Dataset
- **Synthetic ground truth**: closed-form potential-flow solutions (e.g., flow around a cylinder, Joukowski airfoils) and solutions from the AERO-01 panel-method solver.
- **Airfoil geometry**: NACA 4-digit airfoils generated analytically; UIUC Airfoil Coordinates Database for other shapes.
- **Optional reference**: published CFD/experimental results for standard validation cases, cited rather than copied.
- No proprietary data required; the physics residuals themselves provide most of the training signal.

## Tech Stack
- Python, PyTorch (autograd is essential for PDE residual computation)
- DEAP or the same evolutionary algorithm implementation used in the RUL Prediction project, for architecture search
- Matplotlib/Plotly for flow field visualizations, error contour plots
- The Aerodynamics From Scratch panel-method solver, reused as ground truth for validation

## Repository Structure
```
pinn-cfd-aerodynamics/
  src/
    pinn_laplace_2d/
    pinn_airfoil_flowfield/
    pinn_euler_2d/            # exploratory
    neuroevolution_search/
  validation/                 # comparison against Aerodynamics From Scratch panel-method solver
  notebooks/
  docs/
  tests/
```

## Execution Plan
- [x] **Phase 1 — PINN for 2D Laplace equation**: implement a basic PINN solving potential flow around a cylinder, validate against the closed-form analytical solution.
- [x] **Phase 2 — Airfoil flow field PINN**: extend to a parameterized airfoil family, train on a range of shapes/angles of attack, validate against the panel-method solver from Aerodynamics From Scratch.
- [x] **Phase 3 — Neuroevolutionary architecture search**: apply the thesis's neuroevolution methodology to optimize the PINN's architecture and loss weighting for the airfoil flow field problem.
- [x] **Phase 4 — Surrogate speed benchmarking**: measure and document PINN inference speed vs. panel-method solver speed across many repeated shape/condition evaluations.
- [x] **Phase 5 (exploratory) — Extension toward 2D Euler flow**: attempt a PINN formulation for the steady 2D Euler equations on a simplified case, documenting both successes and the genuine difficulties (shock capturing, training stability) — an honest account of the limits is itself valuable, not a weakness.
- [x] **Phase 6 — Documentation and publishing**: write up methodology, results, and an honest discussion of where PINNs help (smooth, well-posed problems, surrogate speed) and where classical methods still win.

## Validation Strategy
- Phase 1 validated against closed-form potential flow solution (exact benchmark available).
- Phase 2 validated against the Aerodynamics From Scratch panel-method solver across a held-out set of airfoil shapes/angles of attack not used in training.
- Report both accuracy (error vs. panel method) and computational cost (training time + inference time) transparently, including failure cases.

## Deliverables
- Public repository with documented PINN solvers and neuroevolution architecture search
- A written comparison report: PINN vs. classical panel-method solver, accuracy and speed trade-offs
- Visualizations: flow field predictions vs. ground truth, error contours, training convergence curves

## Portfolio Differentiators
- Directly and visibly extends your specific academic research contribution (neuroevolution applied to PINNs) into a new, industry-relevant domain — this is a narrative no other candidate can replicate without having done the same thesis.
- Surrogate-model CFD acceleration is an active, real topic in aerospace R&D (Boeing, Airbus, NASA all publish in this space), which makes this project a strong, specific talking point in technical interviews with OEMs.

## Possible Extensions
- Apply the same PINN methodology to the propulsion cycle equations from Propulsion From Scratch, as a physics-informed surrogate for engine performance
- Explore uncertainty quantification in the PINN predictions (ensemble or Bayesian PINNs) as a bridge back to reliability engineering
