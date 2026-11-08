# Lesson 6 · Verification, validation, and when to use a PINN

> Five lessons produced numbers: errors, training times, speed-ups. This lesson asks how far to
> trust them, using the bugs this project actually caught along the way. Then it uses the
> evidence of all five phases to answer the question a reader came with: when is a PINN the
> right tool for an aerodynamics problem?

**Code:** [`tests/`](../../tests/) ·
**Report:** [`docs/comparison-report.md`](../comparison-report.md) ·
**Notebook:** [`06_compute_budget.ipynb`](../../notebooks/06_compute_budget.ipynb) ·
**Previous:** [Lesson 5](05-toward-compressible-flow.md) · **Course index:** [README](README.md)

## Learning objectives

1. Distinguish verifying code, verifying a solution and validating a model, and say which of
   them this project did.
2. Build a test hierarchy for a PINN code, where every test checks a closed-form or cited
   result.
3. Explain how a PINN's error differs from a mesh-based solver's, and what that changes in how
   results are reported.
4. Report compute honestly: error, time per stage, steps to converge, and the baseline.
5. Decide, from evidence, when a PINN is the right tool.

---

## 1. Three questions

Roache (1998) and Oberkampf & Roy (2010) separate three questions:

- **Code verification:** does the code solve the equations it claims to? This is checked
  against exact solutions.
- **Solution verification:** how large is the numerical error of *this* solution? For a mesh
  solver, refine and extrapolate. For a PINN, compare with a reference or check the error's
  spread over seeds and budgets.
- **Validation:** do the equations describe reality? This is checked against experiment.

This project verified code and solutions. It did **not** validate. Potential flow and the Euler
equations are models: they leave out viscosity, so no PINN here predicts stall or drag. The
panel method used as a reference is itself a model; AERO-01 validated it against NACA wind-tunnel
data, within the limits of inviscid theory. When this course says "the PINN matches the panel
method to 1 %", it means one model's numerical solution matches another's, not that either
matches the air.

## 2. The test hierarchy

86 tests, all on the CPU in about 35 s, each asserting against a closed-form result or a cited
value:

| Level | Examples | File (tests) |
|---|---|---|
| Building blocks | autograd Laplacian of harmonic functions is zero; samplers stay in their domains; steps-to-target and plateau on hand-made histories; the budget record replaces a rerun's row | `test_core.py` (13) |
| Exact solutions | the cylinder flow satisfies Laplace and its boundary conditions; Ringleb flow satisfies the Euler equations to 1e-6; the M = 2 shock matches NACA 1135 and conserves fluxes to 1e-12 | `test_laplace_cylinder.py` (12), `test_euler.py` (12) |
| Components against references | the Joukowski slit lies inside every section and its log radius is harmonic, tending to $\ln r$; the pressure-lift integral reproduces AERO-01's own to 1e-6; Newton agrees with Brent's method | `test_airfoil_flowfield.py` (21), `test_surrogate.py` (5) |
| Algorithms against hand-checkable cases | non-dominated sorting and crowding distance on fronts from Deb et al. (2002); genomes stay in their domains | `test_neuroevolution.py` (9) |
| Short seeded training | a small cylinder PINN trained for seconds on the CPU reaches a stated bound, with the reason for the bound written next to it | `test_laplace_cylinder.py`, `test_airfoil_flowfield.py` |
| Documentation | GitHub math renders: no constructs that Markdown mangles inside inline math | `test_docs.py` (14) |

The training tests are the delicate ones. Their bounds state what a short run is expected to
reach, not what the last run happened to reach, and they never get loosened to make a failing
model pass.

## 3. What the tests and checks caught

The value of a test hierarchy shows in what it catches. These are real defects from this
project, in the order they appeared:

| Defect | How it showed | Lesson |
|---|---|---|
| Panel-method field error was NaN | evaluation points at $r = 1$ fell on polygon nodes, where the panel field is singular | evaluate references where they are valid ($r \ge 1.05$) and score the PINN on the same points |
| L-BFGS stopped after a fraction of its iterations | PyTorch's default `max_eval` ended each chunk early, which looked like convergence | read the optimiser's defaults |
| AERO-01's pressure lift disagreed slightly with the same integral here | AERO-01 normalises by the contour's x-extent, not the chord | when two codes disagree, check the definitions first |
| NACA 0012 lay outside the family box | it has $p = 0$; the network was trained on $0.3 \le p \le 0.5$ | symmetric sections need a $p$ inside the box ($p$ has no effect when $m = 0$) |
| A test stand-in returned half the surface velocity | it evaluated the panel method's field velocity *on* the surface, where a vortex sheet gives the average of the inside (zero) and outside values | test the test: its stand-in must reproduce a known number first |
| The surrogate benchmark silently chose the wrong network | it compared a `test_error` key that did not exist, so both values were infinite | fail loudly on missing keys; and choose models on validation, never test, error |
| The first benchmark flattered the PINN by 20× | the baseline was far more accurate than the PINN | compare at equal accuracy (Lesson 4) |
| A Newton test asserted Blasius on a non-streamline contour | pressure alone gives the force only on a streamline; the momentum flux was missing | a test's physics needs checking as much as the code's |
| The compute record flagged clean runs as "dirty" | the runs rewrote the tracked CSV themselves | the provenance column must not be triggered by the run's own output |
| A smoke test overwrote real results | the surrogate's `--quick` mode wrote the real table and figures | quick modes write to scratch, always |

None of these would have shown up as a wrong-looking plot. Each surfaced because a number was
compared with an independent one.

## 4. How a PINN's error is different

A mesh solver's error has structure: it falls at a known order as the mesh is refined, and that
order is itself a verification test. Lesson 1's panel method showed first-order convergence on
the cylinder, as expected. A PINN offers no such handle:

- **No order of accuracy.** More collocation points or a bigger network do not reduce the error
  at a predictable rate. Lesson 3's search found the smaller network more accurate.
- **The optimiser is part of the method.** The same network and loss reach 1e-2 error in 400
  steps with hard constraints and 5600 with soft ones. In every phase, most of the final
  accuracy came in the L-BFGS stage.
- **Seeds matter.** Three seeds of the Phase 2 configuration gave 0.98 % to 1.51 % error. A
  single-seed number has an error bar you have not measured.
- **The residual is not the error.** In notebook 1 the residual falls from the first step while
  the velocity error *rises* from 0.17 to 0.37, then collapses. A PINN must be scored against a
  reference or an exact solution, never by its own loss.
- **Data splits matter.** A parametric PINN is a learned model. Lessons 2–4 kept a test set the
  training and the search never saw, and chose models on validation error only.

So report a PINN result with:
- the error against a reference, on held-out cases;
- its spread over seeds;
- the budget it took: time per optimiser stage and steps to a stated target;
- whether it had plateaued.

The compute record in this project does that for every run:
[`validation/results/README.md`](../../validation/results/README.md) defines the columns, and
[notebook 6](../../notebooks/06_compute_budget.ipynb) summarises them.

## 5. Reporting cost honestly

Lesson 4 showed how much the choice of baseline changes a speed-up: 52× against a 200-panel
solve, 2.5× against a 16-panel solve of the same accuracy. McGreivy & Hakim (2024) surveyed
machine-learning papers on fluid PDEs and found the same pattern widespread: weak baselines and
selective reporting inflate the claimed gains. A fair report states:

- the baseline at **equal accuracy**, or both accuracy and time side by side;
- the **training cost**, and any search that produced the network, with the break-even count;
- **what was timed**: the whole pipeline (here contour building took 85 % of the PINN's time),
  synchronised and warmed up;
- the **hardware and its settings**. Here a GPU was capped at 1000 MHz, and at this network
  size a step is limited by kernel launches, so time follows depth, not parameter count.

## 6. When to use a PINN

The evidence of the five phases, sorted:

| Situation | PINN? | Evidence from this project |
|---|---|---|
| Smooth, well-posed problem, one solve | **No**: a classical solver is cheaper and more accurate | the 512-panel cylinder costs 53 ms; the PINN, 3 minutes of training |
| Smooth problem, exact boundary conditions you can hard-code | Feasible, and accurate | cylinder with hard constraints: 7.7e-5 error |
| A *family* of problems queried millions of times | **Yes**, if the solver is expensive | one network for the NACA family; break-even 2–3 million queries against an equal-accuracy panel method, far fewer against an expensive solver |
| Inverse design: derivatives with respect to inputs | **Yes**: derivatives come free | Newton found the incidence for a target lift in 4 batched evaluations |
| Sparse data plus known physics | **Yes**: physics fills the gaps the data leaves | Phase 2: the data-only network's surface Cp RMS was 0.33, with physics 0.12 |
| High accuracy required (4+ digits) | **No** | the best airfoil network has 0.8 % velocity error and 0.03 lift error; 200 panels give lift to 5e-4 |
| Discontinuities: shocks, contact surfaces | **No**, not without special methods | the Mach 2 wedge stalled at 5 % error with a smeared, misplaced shock |
| Outside the trained parameter range | **No** | lift errors up to 0.37 just outside the box |

In one sentence: a PINN is worth its training cost when the problem is smooth, the same family
of problems is solved many times or differentiated, and two or three digits are enough. For a
one-off high-accuracy solve, or a flow with shocks, a classical method still wins. This matches
the broader assessment of Grossmann et al. (2024), who compared PINNs with finite elements on
standard PDEs.

## 7. Common pitfalls

- **Calling verification validation.** Matching a panel method is not matching an experiment.
- **Scoring a PINN by its loss.** Use an independent reference.
- **One seed.** Measure the spread, or say it was not measured.
- **Choosing models on the test set.** It leaks, and the test error stops being a test.
- **Over-accurate or under-specified baselines.** Compare at equal accuracy and say what was
  timed.
- **Hiding failures.** A shock the network cannot capture is a result. Lesson 5 is the most
  useful lesson in this course for anyone planning compressible PINN work.

## 8. Exercises

1. Classify each of these as code verification, solution verification or validation: (a) the
   Ringleb test; (b) three seeds of the Phase 2 network; (c) AERO-01's NACA 2412 lift against
   wind-tunnel data; (d) the PINN against 400 panels on the test set.
2. Why can a PINN's error not be extrapolated to zero the way a mesh solver's can?
3. Notebook 1's error rose while the residual fell. Give a mechanism, using the hard-constraint
   ansatz, by which a smaller residual can mean a worse velocity field.
4. A paper reports a PINN surrogate 1000× faster than CFD. List the four things you would check
   before believing it.
5. Using the decision table, would you train a PINN to predict the lift of a fixed wing at 20
   flight conditions? Of 10⁶ wing shapes in an optimisation? Of a transonic airfoil with a
   shock?

<details>
<summary>Answers</summary>

1. (a) Code verification: an exact solution checked against the equations. (b) Solution
   verification: the spread of the numerical result. (c) Validation: model against experiment.
   (d) Solution verification against another model's solution, which is not validation.
2. Mesh refinement reduces the error at a known rate $h^p$, so two meshes give an
   extrapolation (Richardson). A PINN's error depends on the optimiser, the seed and the
   network, with no asymptotic rate in any of them, so there is no sequence to extrapolate.
3. The ansatz fixes the boundary values. A network that first makes $d N$ nearly harmonic near
   the outer boundary, where most points are, can lower the mean-square residual while the
   field near the body, where the velocity is largest, is still wrong. The residual is an
   average over points; the velocity error weights the body region more. Many other mechanisms
   are possible; the point is that the two measure different things.
4. The baseline's accuracy compared with the surrogate's; whether training (and data
   generation) is charged; what exactly was timed, and on what hardware; and the error on cases
   held out from training and from model selection.
5. Twenty conditions: no, solve them directly. 10⁶ shapes in an optimisation: yes if the solver
   is expensive and two digits suffice, and use the gradients. Transonic with a shock: not with
   a plain PINN (Lesson 5).

</details>

## Key takeaways

- This project verified code and solutions against exact and panel-method references; it did
  not validate against experiment, and it says so.
- Every test checks a closed-form or cited result. The defects it caught were found by
  comparing numbers, not by looking at plots.
- A PINN's error has no order of accuracy and depends on the optimiser and the seed: report it
  against a reference, over seeds, with its budget.
- Report cost at equal accuracy, charge the training, and say what was timed.
- PINNs fit smooth parametric problems queried many times or differentiated, where two or
  three digits suffice. Classical methods still win on one-off high-accuracy solves and on
  shocks.

## Further reading

- P. J. Roache, *Verification and Validation in Computational Science and Engineering*,
  Hermosa, 1998.
- W. L. Oberkampf & C. J. Roy, *Verification and Validation in Scientific Computing*,
  Cambridge, 2010.
- AIAA, *Guide for the Verification and Validation of Computational Fluid Dynamics Simulations*,
  G-077-1998.
- G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang & L. Yang,
  "Physics-informed machine learning", *Nat. Rev. Phys.* 3, 2021.
- N. McGreivy & A. Hakim, "Weak baselines and reporting biases lead to overoptimism in machine
  learning for fluid-related partial differential equations", *Nat. Mach. Intell.* 6, 2024.
- T. G. Grossmann, U. J. Komorowska, J. Latz & C.-B. Schönlieb, "Can physics-informed neural
  networks beat the finite element method?", *IMA J. Appl. Math.* 89, 2024.
