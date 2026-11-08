"""Generate and execute the demo notebooks from the cell sources below.

Run with ``uv run python notebooks/build_notebooks.py`` (or name notebooks to rebuild only
those). Keeping the sources here, instead of hand-editing .ipynb JSON, makes the notebooks
reviewable in diffs and reproducible. They load the trained networks saved in
``validation/results/checkpoints/`` and run on one CPU thread, so a rebuild on the same machine
gives byte-identical files; nothing in them is timed live (timings come from the
compute-budget record).
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

SETUP = """\
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from pinncfd import io, plotting as P

torch.set_num_threads(1)  # one thread: reductions in a fixed order, reproducible outputs
P.use_style()
DEG = np.pi / 180
ROOT = Path("..").resolve()
RESULTS = ROOT / "validation" / "results"
CK = RESULTS / "checkpoints"
DEVICE = "cpu\""""

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_laplace_cylinder.ipynb": [
        (
            "md",
            """\
# 1 · A PINN for potential flow past a cylinder

The network learns the stream function $\\psi$ on the annulus $1 \\le r \\le 5$, with
$\\nabla^2\\psi = 0$, the body a streamline, and the exact solution as data on $r = 5$. With hard
constraints the boundary conditions hold for any network output, and only the Laplace residual
is trained. See [Lesson 1](../docs/lessons/01-pinns-for-potential-flow.md) and
[docs/pinn_laplace_2d.md](../docs/pinn_laplace_2d.md).""",
        ),
        (
            "code",
            SETUP
            + """
from pinncfd.pinn_laplace_2d import exact
from pinncfd.pinn_laplace_2d.cylinder import CylinderPINN, CylinderProblem
from pinncfd.training import TrainConfig, train""",
        ),
        ("md", "The exact solution's surface pressure integrates to $c_l = \\Gamma$:"),
        (
            "code",
            """\
th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
for gamma in (0.0, np.pi, 2 * np.pi):
    print(f"Gamma = {gamma:.4f}:  cl from Cp = {exact.cl_from_cp(th, exact.surface_cp(th, gamma)):.6f}")""",
        ),
        ("md", "The trained networks (seed 0), evaluated on a 64 × 128 polar grid:"),
        (
            "code",
            """\
models = {}
for case in ("psi_soft", "psi_hard", "psi_hard_lifting", "phi_soft"):
    model, _ = io.load_checkpoint(CK / f"phase1_{case}.pt", CylinderPINN, DEVICE)
    models[case] = model
    m = CylinderProblem(model, n_interior=16, device=DEVICE).evaluate(model)
    print(f"{case:17s} velocity error {m['error']:.2e}   max |dCp| {m['cp_max_abs']:.2e}")""",
        ),
        (
            "code",
            """\
model = models["psi_hard_lifting"]
prob = CylinderProblem(model, n_interior=16, device=DEVICE)
th = np.linspace(0, 2 * np.pi, 361)
_, u, v = prob.predict(model, np.column_stack([np.cos(th), np.sin(th)]))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.degrees(th), exact.surface_cp(th, model.gamma), "--", color=P.REFERENCE, label="exact")
ax.plot(np.degrees(th), 1 - u**2 - v**2, color=P.SERIES[0], label="PINN, hard BCs")
ax.invert_yaxis()
ax.set_xlabel("θ [deg]"); ax.set_ylabel("Cp"); ax.set_title("Lifting cylinder, Γ = 2π"); ax.legend();""",
        ),
        ("md", "Convergence of the full runs, from the saved histories:"),
        (
            "code",
            """\
hist = json.loads((RESULTS / "phase1_histories.json").read_text())
fig, ax = plt.subplots(figsize=(7, 4))
for i, case in enumerate(("psi_soft", "psi_hard", "phi_soft")):
    P.convergence_plot(ax, hist[f"{case}/0"], label=case, color=P.SERIES[i],
                       adam_steps=5000 if i == 0 else None)
ax.set_ylabel("velocity error"); ax.legend();""",
        ),
        (
            "md",
            "A short training run on the CPU, with the network of the full runs (4 × 50 tanh), "
            "1024 points, 400 Adam steps and 100 L-BFGS iterations. Watch the two columns: the "
            "loss (the Laplace residual) falls from the start, but the velocity error first "
            "*rises*, from 0.17 to 0.37, before it collapses. The full run in the plot above did "
            "the same (0.38 at step 100, 0.012 at step 400). A small residual is not yet a "
            "correct solution: the error has to be measured against a reference.",
        ),
        (
            "code",
            """\
torch.manual_seed(0)
model = CylinderPINN(hard=True, mlp=dict(width=50, depth=4, seed=0))
prob = CylinderProblem(model, n_interior=1024, n_boundary=128, device=DEVICE)
cfg = TrainConfig(adam_steps=400, lbfgs_iters=100, eval_every=50, resample_every=0, seed=0)
res = train(model, prob, cfg, DEVICE)
for h in res.history:
    print(f"step {h['step']:4d} ({h['stage']:6s})  loss {h['loss']:.3e}   velocity error {h['error']:.3e}")""",
        ),
    ],
    "02_airfoil_family.ipynb": [
        (
            "md",
            """\
# 2 · One network for a family of airfoils

$\\psi(x, y; m, p, t, \\alpha)$ for every NACA 4-digit section with $m \\le 0.06$,
$0.3 \\le p \\le 0.5$, $0.08 \\le t \\le 0.18$ and $-4° \\le \\alpha \\le 8°$. The network sees
inverse-Joukowski coordinates; the lift is measured from the predicted field. See
[Lesson 2](../docs/lessons/02-airfoil-flow-fields.md) and
[docs/pinn_airfoil_flowfield.md](../docs/pinn_airfoil_flowfield.md).""",
        ),
        (
            "code",
            SETUP
            + """
from pinncfd.external.geometry import naca, panel_solution
from pinncfd.pinn_airfoil_flowfield import AirfoilPINN, build_reference, family
from pinncfd.pinn_airfoil_flowfield.problem import evaluate, lift_from_circulation, predict_velocity

model, _ = io.load_checkpoint(CK / "phase2_physics_data.pt", AirfoilPINN, DEVICE)""",
        ),
        ("md", "Held-out test sections, against a 400-panel AERO-01 solution:"),
        (
            "code",
            """\
cases = [family._naca(code, a) for code, a in (("0012", 4.0), ("2412", 4.0), ("4415", 7.0))]
ref = build_reference(cases, n_field=256)
s = evaluate(model, ref, DEVICE, per_case=True)
for i, c in enumerate(cases):
    print(f"{c.name:20s} velocity error {s['error'][i]:.4f}  cl panel {s['cl_ref'][i]:.3f}  "
          f"PINN (circulation) {s['cl_circulation'][i]:.3f}  (pressure) {s['cl_pressure'][i]:.3f}")""",
        ),
        (
            "code",
            """\
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
xs = ref.surf_xy[:, 0].reshape(len(cases), -1)
cp_ref = ref.surf_cp.reshape(len(cases), -1)
for ax, i in zip(axes, range(len(cases))):
    ax.plot(xs[i], cp_ref[i], "--", color=P.REFERENCE, lw=1, label="panel, 400")
    ax.plot(xs[i], s["cp"][i], color=P.SERIES[0], lw=1, label="PINN")
    ax.set_title(cases[i].name, fontsize=10); ax.set_xlabel("x / c")
axes[0].invert_yaxis(); axes[0].set_ylabel("Cp"); axes[0].legend();""",
        ),
        (
            "md",
            "The speed field around a cambered section, predicted for a shape and incidence "
            "the network never saw with data:",
        ),
        (
            "code",
            """\
from pinncfd.external.geometry import inside
m, p, t, a = 0.035, 0.42, 0.13, 5.5 * DEG
X, Y = np.meshgrid(np.linspace(-0.6, 1.6, 221), np.linspace(-0.7, 0.7, 141))
xy = np.column_stack([X.ravel(), Y.ravel()])
u, v = predict_velocity(model, xy, np.tile([m, p, t, a], (len(xy), 1)), DEVICE)
speed = np.where(inside(m, p, t, xy), np.nan, np.hypot(u, v)).reshape(X.shape)
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.contourf(X, Y, speed, levels=np.linspace(0, 1.8, 19), cmap="Blues")
ax.fill(*naca(m, p, t).coordinates(200).T, color=P.TEXT_MUTED)
ax.set_aspect("equal"); ax.grid(False); fig.colorbar(im, label="|V| / V∞")
ax.set_title("m = 3.5 %, p = 0.42, t = 13 %, α = 5.5°");""",
        ),
        ("md", "Lift curves of three sections, PINN against the panel method:"),
        (
            "code",
            """\
alphas = np.linspace(-4, 8, 13)
fig, ax = plt.subplots(figsize=(7, 4))
for i, code in enumerate(("0012", "2412", "4415")):
    q = np.array([family._naca(code, al).params for al in alphas])
    ax.plot(alphas, lift_from_circulation(model, q, DEVICE), color=P.SERIES[i], label=f"{code} PINN")
    ax.plot(alphas, [panel_solution(*r[:3], r[3], 200).cl for r in q], "--", color=P.SERIES[i], lw=1)
ax.set_xlabel("α [deg]"); ax.set_ylabel("cl"); ax.set_title("solid: PINN, dashed: panel method"); ax.legend();""",
        ),
    ],
    "03_neuroevolution.ipynb": [
        (
            "md",
            """\
# 3 · Neuroevolution of a PINN

NSGA-II over twelve genes (architecture, learning rate, collocation count, five loss weights),
minimising validation error and parameter count. See
[Lesson 3](../docs/lessons/03-neuroevolution.md) and
[docs/neuroevolution_search.md](../docs/neuroevolution_search.md).""",
        ),
        (
            "code",
            SETUP
            + """
from pinncfd.neuroevolution_search import genome as G
from pinncfd.neuroevolution_search.nsga import crowding_distance, fast_non_dominated_sort""",
        ),
        ("md", "Non-dominated sorting and crowding distance on a small hand-made set:"),
        (
            "code",
            """\
pts = [(0.03, 6000), (0.02, 9000), (0.03, 8000), (0.05, 4500), (0.06, 9000), (0.025, 7000)]
fronts = fast_non_dominated_sort(pts)
for k, f in enumerate(fronts):
    d = crowding_distance([pts[i] for i in f])
    print(f"front {k}: " + ", ".join(f"{pts[i]} d={di:.2f}" for i, di in zip(f, d)))""",
        ),
        ("md", "The 40 genomes of the search, and its final front:"),
        (
            "code",
            """\
state = json.loads((RESULTS / "phase3_search.json").read_text())
ev = state["evaluated"]
err = np.array([e["fitness"]["error"] for e in ev])
npar = np.array([e["fitness"]["n_params"] for e in ev])
front = fast_non_dominated_sort([(e["fitness"]["error"], e["fitness"]["n_params"]) for e in state["population"]])[0]
fig, ax = plt.subplots(figsize=(7, 4.5))
sc = ax.scatter(npar, err, c=[e["generation"] for e in ev], cmap="Blues", vmin=-1, vmax=4, edgecolors=P.TEXT_MUTED, lw=0.3)
fp = sorted((state["population"][i]["fitness"]["n_params"], state["population"][i]["fitness"]["error"]) for i in front)
ax.plot(*zip(*fp), "o-", mfc="none", color=P.SERIES[1], label="final front")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("parameters"); ax.set_ylabel("validation error (proxy)")
fig.colorbar(sc, label="generation"); ax.legend();
for i in sorted(front, key=lambda i: state["population"][i]["fitness"]["error"]):
    ind = state["population"][i]
    print(f"{ind['fitness']['error']:.4f}  {ind['fitness']['n_params']:6d}  {G.describe(ind['genome'])}")""",
        ),
        (
            "md",
            "What the genes did (with the caveats of Lesson 3: one seed, genes shared by descent):",
        ),
        (
            "code",
            """\
for gene in ("activation", "fourier_features", "depth"):
    vals = sorted({e["genome"][gene] for e in ev}, key=str)
    print(gene + ": " + ", ".join(
        f"{v}: median {np.median([e['fitness']['error'] for e in ev if e['genome'][gene] == v]):.3f} "
        f"(n={sum(e['genome'][gene] == v for e in ev)})" for v in vals))""",
        ),
        ("md", "The champion against the hand-tuned network at the full budget (seed 0):"),
        (
            "code",
            """\
h2 = json.loads((RESULTS / "phase2_histories.json").read_text())["physics_data/0"]
h3 = json.loads((RESULTS / "phase3_histories.json").read_text())["champion/0"]
fig, ax = plt.subplots(figsize=(7, 4))
P.convergence_plot(ax, h2, label="Phase 2, hand-tuned", color=P.SERIES[0], adam_steps=4000)
P.convergence_plot(ax, h3, label="Phase 3 champion", color=P.SERIES[1])
ax.set_ylabel("validation velocity error"); ax.legend();""",
        ),
    ],
    "04_surrogate_benchmark.ipynb": [
        (
            "md",
            """\
# 4 · The PINN as a surrogate

Batched lift, Newton on the incidence with autograd derivatives, and the cost model. Timings
come from the benchmark's record (`validation/results/compute_budget.csv`, phase4), not from
this notebook, so it rebuilds identically. See
[Lesson 4](../docs/lessons/04-surrogates-in-a-design-loop.md) and
[docs/surrogate.md](../docs/surrogate.md).""",
        ),
        (
            "code",
            SETUP
            + """
from pinncfd.external.geometry import naca, solve_panels
from pinncfd.pinn_airfoil_flowfield import AirfoilPINN, family
from pinncfd.surrogate import break_even, cl_circulation, cl_pressure, solve_alpha

model, _ = io.load_checkpoint(CK / "phase3_champion_s0.pt", AirfoilPINN, DEVICE)""",
        ),
        ("md", "Lift of 200 random sections in one call, against 400 and 16 panels:"),
        (
            "code",
            """\
q = family.uniform(200, np.random.default_rng(4))
def panel_cl(n):
    return np.array([solve_panels(naca(m, p, t).coordinates(n), a).cl for m, p, t, a in q])
ref, coarse = panel_cl(400), panel_cl(16)
for name, cl in (("PINN, pressure", cl_pressure(model, q, DEVICE)),
                 ("PINN, circulation", cl_circulation(model, q, DEVICE)),
                 ("panel, 16 panels", coarse)):
    print(f"{name:18s} mean |cl error| {np.abs(cl - ref).mean():.4f}")""",
        ),
        ("md", "The incidence for $c_l = 0.8$ on the six test sections, all at once by Newton:"),
        (
            "code",
            """\
q0 = np.array([[*family._naca(code, 0.0).params[:3], 0.0] for code in family.TEST_SECTIONS])
alpha, cl, n_eval = solve_alpha(model, q0, 0.8, DEVICE)
print(f"{n_eval} batched evaluations")
for code, row, a in zip(family.TEST_SECTIONS, q0, alpha):
    check = solve_panels(naca(*row[:3]).coordinates(200), a).cl
    print(f"NACA {code}: alpha = {a / DEG:6.3f} deg, panel cl there = {check:.3f}")""",
        ),
        ("md", "Timings and break-even counts from the benchmark record:"),
        (
            "code",
            """\
rows = {r["case"]: r for r in io.read_budget() if r["phase"] == "phase4"}
t = {k: float(r["time_total_s"]) for k, r in rows.items()}
per_pinn = t["pinn_gpu_x10000"] / 1e4
train_s = float(next(r for r in io.read_budget() if r["phase"] == "phase3" and r["case"] == "champion" and r["seed"] == "0")["time_total_s"])
# 16 panels: the panel-count sweep; 200 panels: the 10^4-evaluation timing (as in Lesson 4)
for n_pan, t_pan in ((16, t["panel_N16"]), (200, t["panel_x10000"] / 1e4)):
    print(f"{n_pan:3d} panels: {1e3 * t_pan:.2f} ms/case, PINN {1e6 * per_pinn:.0f} us/case, "
          f"break-even {break_even(train_s, t_pan, per_pinn):,.0f} evaluations")""",
        ),
        (
            "code",
            """\
n = np.logspace(0, 7, 50)
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(n, n * t["panel_x10000"] / 1e4, "--", color=P.REFERENCE, label="panel, 200")
ax.loglog(n, n * t["panel_N16"], "--", color=P.TEXT_MUTED, label="panel, 16 (as accurate as the PINN)")
ax.loglog(n, train_s + n * per_pinn, color=P.SERIES[0], label="PINN, training + inference")
ax.set_xlabel("lift evaluations"); ax.set_ylabel("total time [s]"); ax.legend();""",
        ),
    ],
    "05_euler.ipynb": [
        (
            "md",
            """\
# 5 · Toward compressible flow: Euler PINNs

Two exact solutions of the steady Euler equations, and what PINNs make of them: smooth Ringleb
flow, and a Mach 2 flow over a 10° wedge with an attached oblique shock. See
[Lesson 5](../docs/lessons/05-toward-compressible-flow.md) and
[docs/pinn_euler_2d.md](../docs/pinn_euler_2d.md).""",
        ),
        (
            "code",
            SETUP
            + """
from pinncfd.pinn_euler_2d import EulerPINN, RinglebProblem, WedgeProblem
from pinncfd.pinn_euler_2d.exact import oblique_shock, ringleb, wedge_field""",
        ),
        ("md", "The oblique shock at $M = 2$, $\\theta = 10°$ (NACA Report 1135 charts):"),
        (
            "code",
            """\
s = oblique_shock(10 * DEG, 2.0)
print(f"beta = {s['beta'] / DEG:.2f} deg, p2/p1 = {s['p_ratio']:.4f}, "
      f"rho2/rho1 = {s['rho_ratio']:.4f}, M2 = {s['mach2']:.4f}")""",
        ),
        ("md", "Ringleb flow: the Mach number, with a supersonic pocket next to the inner wall:"),
        (
            "code",
            """\
K, QN = np.meshgrid(np.linspace(0.7, 1.2, 60), np.linspace(0, 1, 120))
Q = 0.5 + (K - 0.5) * QN * (1 - 1e-9)
fig, ax = plt.subplots(figsize=(6, 5))
for b in (1, -1):
    x, y, rho, u, v, p = ringleb(Q, K, b)
    cs = ax.contourf(x, y, np.hypot(u, v) / np.sqrt(1.4 * p / rho), levels=np.linspace(0.3, 1.5, 13), cmap="Blues")
    ax.contour(x, y, np.hypot(u, v) / np.sqrt(1.4 * p / rho), levels=[1.0], colors=P.SERIES[1])
ax.set_aspect("equal"); ax.grid(False); fig.colorbar(cs, label="Mach"); ax.set_title("Ringleb flow");""",
        ),
        ("md", "The trained networks (seed 0), against the exact solutions:"),
        (
            "code",
            """\
for case in ("ringleb", "wedge_plain", "wedge_viscous", "wedge_weighted"):
    model, _ = io.load_checkpoint(CK / f"phase5_{case}.pt", EulerPINN, DEVICE)
    cls = RinglebProblem if case == "ringleb" else WedgeProblem
    m = cls(model, n_interior=64, n_boundary=32, device=DEVICE).evaluate(model)
    extra = f", shock angle {m['beta_deg']:.2f} deg" if "beta_deg" in m else ""
    print(f"{case:15s} state error {m['error']:.3e}" + extra)""",
        ),
        (
            "code",
            """\
g = np.linspace(0, 1, 201)
X, Y = np.meshgrid(g, g)
pts = np.column_stack([X.ravel(), Y.ravel()])
below = pts[:, 1] < pts[:, 0] * np.tan(10 * DEG)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
model, _ = io.load_checkpoint(CK / "phase5_wedge_viscous.pt", EulerPINN, DEVICE)
with torch.no_grad():
    rho_pinn = model(torch.tensor(pts, dtype=torch.float32))[:, 0].numpy()
for ax, rho, title in ((axes[0], wedge_field(*pts.T, 10 * DEG, 2.0)[:, 0], "exact"),
                       (axes[1], rho_pinn, "PINN, artificial viscosity")):
    im = ax.pcolormesh(X, Y, np.where(below, np.nan, rho).reshape(X.shape), cmap="Blues", vmin=0.9, vmax=1.6, shading="auto")
    ax.set_aspect("equal"); ax.grid(False); ax.set_title(title); ax.set_xlabel("x")
axes[0].set_ylabel("y"); fig.colorbar(im, ax=axes, label="ρ / ρ∞");""",
        ),
    ],
    "06_compute_budget.ipynb": [
        (
            "md",
            """\
# 6 · The compute budget

Every training run wrote one row to `validation/results/compute_budget.csv`: its error,
training time per optimiser stage and steps to converge. This notebook summarises it per phase.
Column definitions: [validation/results/README.md](../validation/results/README.md). See
[Lesson 6](../docs/lessons/06-verification-validation-and-when-to-use-a-pinn.md).""",
        ),
        ("code", SETUP),
        (
            "code",
            """\
rows = [r for r in io.read_budget() if r["adam_steps"]]  # training runs only
groups = {}
for r in rows:  # the 40 search genomes form one group
    case = "search (proxy)" if r["case"].startswith("search_") else r["case"]
    groups.setdefault((r["phase"], case), []).append(r)

def steps(r, target):  # None: target not recorded for this phase; -1: never reached
    return json.loads(r["metrics_json"]).get(f"steps_to_{target:.0e}")

print("mean error and training time; median steps to the target over the runs that reached it")
print(f"{'phase':7s} {'case':22s} {'runs':>4s} {'error':>9s} {'time [s]':>9s} {'to 3e-2':>8s} {'to 1e-2':>8s}")
for (phase, case), rs in sorted(groups.items()):
    e = np.mean([float(r["error"]) for r in rs])
    t = np.mean([float(r["time_total_s"]) for r in rs])
    s3 = [steps(r, 3e-2) for r in rs]
    s2 = [steps(r, 1e-2) for r in rs]
    fmt = lambda s: "—" if None in s else "never" if max(s) < 0 else f"{int(np.median([x for x in s if x >= 0]))}"
    print(f"{phase:7s} {case:22s} {len(rs):4d} {e:9.2e} {t:9.0f} {fmt(s3):>8s} {fmt(s2):>8s}")""",
        ),
        (
            "code",
            """\
per_phase = {}
for r in rows:
    per_phase[r["phase"]] = per_phase.get(r["phase"], 0.0) + float(r["time_total_s"])
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(list(per_phase), [v / 60 for v in per_phase.values()], color=P.SERIES[0])
ax.set_xlabel("GPU training time [min], RTX 2060 at 1000 MHz"); ax.invert_yaxis()
for k, v in per_phase.items():
    print(f"{k}: {v / 60:.1f} min over {sum(r['phase'] == k for r in rows)} runs")""",
        ),
    ],
}


def _merge_streams(outputs):
    """Join consecutive stdout/stderr chunks; how the kernel splits them depends on timing."""
    merged = []
    for out in outputs:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and out.get("output_type") == "stream"
            and prev.get("output_type") == "stream"
            and prev.get("name") == out.get("name")
        ):
            prev["text"] += out["text"]
        else:
            merged.append(out)
    return merged


def build(name: str, cells: list[tuple[str, str]]) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        for kind, src in cells
    ]
    for i, cell in enumerate(nb.cells):
        cell.id = f"cell-{i:02d}"  # stable ids keep rebuilds diff-free
    NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(HERE)}},
    ).execute()
    for cell in nb.cells:
        cell.get("outputs", [])[:] = _merge_streams(cell.get("outputs", []))
    path = HERE / name
    nbformat.write(nb, path)
    return path


def main(names: list[str]) -> None:
    for name, cells in NOTEBOOKS.items():
        if not names or name in names or name.split("_")[0] in names:
            print("built", build(name, cells).name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
