"""Phase 4: the airfoil PINN as a surrogate for the panel method, in time and accuracy.

1. Accuracy: lift of 500 random sections of the family (uniform in the box, seed 4) from the
   panel method (200 panels, the baseline) and from the PINN (surface pressure and circulation).
2. Speed: wall time for n = 1 ... 10^4 lift evaluations — the panel method one case at a time,
   the PINN in one batch on the GPU and on the CPU — and the break-even number of evaluations
   once the PINN's training time is charged to it, both against the 200-panel baseline and
   against the coarsest panel method that is at least as accurate as the PINN (a panel-count
   sweep on the same 500 sections).
3. Design loop: (a) the incidence giving cl = 0.8 for each test section: Brent's method on
   panel solutions vs batched Newton on the PINN (d cl / d alpha by autograd through the
   network's alpha input) and vs a batched PINN sweep of the whole cl-alpha curve; (b) a cl
   carpet over camber and thickness.

The surrogate is whichever of the Phase 2 network and the Phase 3 champion (seed 0) has the
lower *validation* velocity error (the test set stays out of model selection). Writes
``validation/results/surrogate_benchmark.md``, rows in
``compute_budget.csv`` and ``docs/figures/surrogate_*.png``.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from pinncfd import io, plotting
from pinncfd.device import get_device
from pinncfd.external.geometry import naca, solve_panels
from pinncfd.nn import n_parameters
from pinncfd.pinn_airfoil_flowfield import AirfoilPINN, family
from pinncfd.surrogate import (
    break_even,
    cl_circulation,
    cl_pressure,
    solve_alpha,
    surface_panels,
    timed,
)

from common import CHECKPOINTS, RESULTS

PHASE = "phase4"
DEG = np.pi / 180
PANELS = 200
N_ACCURACY = 500
SIZES = (1, 10, 100, 1000, 10000)
PANEL_SWEEP = (8, 12, 16, 24, 32, 48, 64, 100, 200)
TARGET_CL = 0.8


def panel_cl(params, n=PANELS):
    return np.array([solve_panels(naca(m, p, t).coordinates(n), a).cl for m, p, t, a in params])


def choose_model(device):
    """The surrogate: the Phase 3 champion (seed 0) if its validation error beats Phase 2's.

    Returns (model, name, training time [s], search time [s], validation error); the search
    time is the whole Phase 3 search when the champion is chosen, else 0."""
    rows = io.read_budget()

    def run(phase, case):
        return next(
            (r for r in rows if r["phase"] == phase and r["case"] == case and r["seed"] == "0"),
            None,
        )

    r2, r3 = run("phase2", "physics_data"), run("phase3", "champion")
    if r3 is not None and float(r3["error"]) < float(r2["error"]):
        path, name, row = CHECKPOINTS / "phase3_champion_s0.pt", "Phase 3 champion", r3
        search_s = sum(
            float(r["time_total_s"])
            for r in rows
            if r["phase"] == "phase3" and r["case"].startswith("search_")
        )
    else:
        path, name, row, search_s = CHECKPOINTS / "phase2_physics_data.pt", "Phase 2 network", r2, 0
    model, _ = io.load_checkpoint(path, AirfoilPINN, device)
    return model.eval(), name, float(row["time_total_s"]), search_s, float(row["error"])


def budget_row(case, error, time_s, n_params, device, notes, metrics):
    row = {
        "phase": PHASE,
        "case": case,
        "seed": 0,
        "error_name": "cl_abs_err_mean",
        "error": error,
        "time_total_s": time_s,
        "n_params": n_params,
        "metrics_json": json.dumps(metrics),
        "notes": notes,
    }
    row.update(io.environment(device))
    io.record_run(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    gpu = get_device()
    sizes = (1, 10) if args.quick else SIZES
    n_acc = 20 if args.quick else N_ACCURACY

    model, name, train_s, search_s, val_err = choose_model(gpu)
    cpu_model, *_ = choose_model("cpu")
    n_par = n_parameters(model)
    print(f"surrogate: {name} ({n_par} parameters, trained in {train_s:.0f} s)", flush=True)

    # ---- 1. accuracy ----------------------------------------------------------------------
    params = family.uniform(n_acc, np.random.default_rng(4))
    cl_ref = panel_cl(params, 400)
    cl_pan = panel_cl(params)
    cl_p = cl_pressure(model, params, gpu)
    cl_c = cl_circulation(model, params, gpu)
    acc = {
        "panel200": np.abs(cl_pan - cl_ref),
        "pinn_pressure": np.abs(cl_p - cl_ref),
        "pinn_circulation": np.abs(cl_c - cl_ref),
    }

    # accuracy and cost of the panel method against its panel count (the equal-accuracy baseline)
    sweep_err, sweep_cost = {}, {}
    q_cost = family.uniform(50 if args.quick else 1000, np.random.default_rng(6))
    for n_pan in PANEL_SWEEP:
        sweep_err[n_pan] = np.abs(panel_cl(params, n_pan) - cl_ref)
        _, t = timed(panel_cl, q_cost, n_pan, repeats=1)
        sweep_cost[n_pan] = t / len(q_cost)
    n_equal = next(
        (n for n in PANEL_SWEEP if sweep_err[n].mean() <= acc["pinn_pressure"].mean()), PANELS
    )

    # ---- 2. speed -------------------------------------------------------------------------
    keys = ("panel", "pinn_gpu_pressure", "pinn_gpu_circulation", "pinn_cpu", "contours")
    timing = {k: {} for k in keys}
    rng = np.random.default_rng(5)
    for n in sizes:
        q = family.uniform(n, rng)
        reps = 1 if n >= 1000 else 3
        _, timing["panel"][n] = timed(panel_cl, q[: min(n, 1000)], repeats=reps)
        if n > 1000:  # the panel method is strictly one case at a time: scale the 1000-case time
            timing["panel"][n] *= n / 1000
        _, timing["pinn_gpu_pressure"][n] = timed(cl_pressure, model, q, gpu, device=gpu)
        _, timing["pinn_gpu_circulation"][n] = timed(cl_circulation, model, q, gpu, device=gpu)
        _, timing["pinn_cpu"][n] = timed(cl_pressure, cpu_model, q, "cpu", repeats=reps)
        _, timing["contours"][n] = timed(surface_panels, q, 120, repeats=reps)
        tm = {k: v[n] for k, v in timing.items()}
        print(
            f"n={n:6d}: panel {tm['panel']:.3g} s, PINN GPU {tm['pinn_gpu_pressure']:.3g} s "
            f"(circulation {tm['pinn_gpu_circulation']:.3g} s), CPU {tm['pinn_cpu']:.3g} s, "
            f"of which contours {tm['contours']:.3g} s",
            flush=True,
        )
    big = max(sizes)
    per_panel = timing["panel"][big] / big
    per_gpu = timing["pinn_gpu_pressure"][big] / big
    per_cpu = timing["pinn_cpu"][big] / big
    n_star_gpu = break_even(train_s, per_panel, per_gpu)
    n_star_cpu = break_even(train_s, per_panel, per_cpu)
    n_star_search = break_even(train_s + search_s, per_panel, per_gpu)
    per_circ = timing["pinn_gpu_circulation"][big] / big
    n_star_eq = break_even(train_s, sweep_cost[n_equal], per_gpu)
    n_star_eq_circ = break_even(train_s, sweep_cost[n_equal], per_circ)

    # ---- 3a. inverse problem: alpha for cl = 0.8 ------------------------------------------
    sections = family.TEST_SECTIONS
    inv = []
    for code in sections:
        c = family._naca(code, 0.0)
        nodes = naca(c.m, c.p, c.t).coordinates(PANELS)
        calls = [0]

        def f(a, nodes=nodes, calls=calls):
            calls[0] += 1
            return solve_panels(nodes, a).cl - TARGET_CL

        t0 = time.perf_counter()
        a_panel = brentq(f, -10 * DEG, 15 * DEG, xtol=1e-6)
        t_panel = time.perf_counter() - t0
        inv.append([code, c, a_panel, calls[0], t_panel])
    alphas = np.linspace(-4, 12, 321) * DEG
    q = np.array([[c.m, c.p, c.t, a] for _, c, *_ in inv for a in alphas])

    def sweep():
        return cl_pressure(model, q, gpu).reshape(len(inv), len(alphas))

    grid, t_pinn = timed(sweep, device=gpu)
    q0 = np.array([[c.m, c.p, c.t, 0.0] for _, c, *_ in inv])
    (a_newton, _, newton_evals), t_newton = timed(
        solve_alpha, model, q0, TARGET_CL, gpu, device=gpu
    )
    inverse_rows = []
    for k, (code, c, a_panel, calls, t_panel) in enumerate(inv):
        a_pinn = float(a_newton[k])
        a_sweep = float(np.interp(TARGET_CL, grid[k], alphas))
        cl_true = solve_panels(naca(c.m, c.p, c.t).coordinates(PANELS), a_pinn).cl
        inverse_rows.append((code, a_panel, calls, t_panel, a_pinn, a_sweep, cl_true))

    # ---- 3b. carpet ------------------------------------------------------------------------
    g = 6 if args.quick else 41
    mm, tt = np.meshgrid(np.linspace(0, 0.06, g), np.linspace(0.08, 0.18, g))
    qc = np.column_stack([mm.ravel(), np.full(mm.size, 0.4), tt.ravel(), np.full(mm.size, 4 * DEG)])
    carpet_panel, t_carpet_panel = timed(panel_cl, qc, repeats=1)
    carpet_pinn, t_carpet_pinn = timed(cl_pressure, model, qc, gpu, device=gpu)
    # ---- record (not for --quick, whose timings are not the benchmark) ---------------------
    record = not args.quick
    for n in sizes if record else ():
        budget_row(
            f"panel_x{n}",
            float(acc["panel200"].mean()),
            timing["panel"][n],
            PANELS,
            "cpu",
            f"{n} lift evaluations, one at a time; >1000 scaled from 1000",
            {"per_case_s": timing["panel"][n] / n},
        )
        budget_row(
            f"pinn_gpu_x{n}",
            float(acc["pinn_pressure"].mean()),
            timing["pinn_gpu_pressure"][n],
            n_par,
            gpu,
            f"{n} lift evaluations in one batch, surface pressure, 120 panels",
            {
                "per_case_s": timing["pinn_gpu_pressure"][n] / n,
                "circulation_s": timing["pinn_gpu_circulation"][n],
            },
        )
        budget_row(
            f"pinn_cpu_x{n}",
            float(acc["pinn_pressure"].mean()),
            timing["pinn_cpu"][n],
            n_par,
            "cpu",
            f"{n} lift evaluations in one batch on the CPU",
            {"per_case_s": timing["pinn_cpu"][n] / n},
        )

    lines = [
        "# Phase 4 — The PINN as a surrogate for the panel method",
        "",
        f"Surrogate: {name}, seed 0 ({n_par} parameters; validation velocity error "
        f"{val_err:.3g}; training time {train_s:.0f} s on the RTX 2060 at 1000 MHz"
        + (f", after a {search_s / 60:.0f}-minute architecture search)" if search_s else ")")
        + f". Chosen on validation error. Baseline: AERO-01 panel method, {PANELS} panels. "
        "Reference for accuracy: 400 panels.",
        "",
        f"## Lift accuracy on {n_acc} random sections of the family",
        "",
        "| Method | Mean abs cl error | 95th percentile | Max |",
        "|---|---|---|---|",
    ]
    labels = {
        "panel200": f"panel method, {PANELS} panels",
        "pinn_pressure": "PINN, surface pressure (120 points)",
        "pinn_circulation": "PINN, circulation (256 points on r = 1.5c)",
    }
    for k, e in acc.items():
        lines.append(
            f"| {labels[k]} | {e.mean():.3g} | {np.percentile(e, 95):.3g} | {e.max():.3g} |"
        )
    lines += [
        "",
        "## Wall time for n lift evaluations",
        "",
        "| n | Panel method (CPU) | PINN, GPU (pressure) | PINN, GPU (circulation) "
        "| PINN, CPU (pressure) | Contours alone (CPU) | Speed-up, GPU |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in sizes:
        lines.append(
            f"| {n} | {timing['panel'][n]:.3g} s | {timing['pinn_gpu_pressure'][n]:.3g} s "
            f"| {timing['pinn_gpu_circulation'][n]:.3g} s | {timing['pinn_cpu'][n]:.3g} s "
            f"| {timing['contours'][n]:.3g} s "
            f"| {timing['panel'][n] / timing['pinn_gpu_pressure'][n]:.3g}× |"
        )
    lines += [
        "",
        f"Per case at n = {big}: panel {1e3 * per_panel:.3g} ms, PINN GPU "
        f"{1e6 * per_gpu:.3g} µs, PINN CPU {1e6 * per_cpu:.3g} µs. Panel times above 1000 "
        "cases are the 1000-case time scaled (the method is strictly sequential). The "
        "pressure-lift PINN times include building each section's 120-panel contour in NumPy "
        "(column 'Contours alone'); the circulation lift needs no contour. GPU times are "
        "synchronised; all times are medians after a warm-up call.",
        "",
        f"**Break-even** (training time charged to the PINN): {n_star_gpu:,.0f} evaluations "
        f"on the GPU, {n_star_cpu:,.0f} on the CPU"
        + (
            f"; {n_star_search:,.0f} on the GPU if the architecture search is charged too."
            if search_s
            else "."
        ),
        "",
        "## The equal-accuracy baseline",
        "",
        f"The PINN's lift is less accurate than the {PANELS}-panel baseline, so the fair "
        "comparison is with the coarsest panel method that is at least as accurate. Panel "
        f"count sweep on the same {n_acc} sections (error against 400 panels; time per case "
        f"over {len(q_cost)} other sections, one at a time):",
        "",
        "| Panels | Mean abs cl error | 95th percentile | Time per case |",
        "|---|---|---|---|",
        *[
            f"| {n_pan} | {sweep_err[n_pan].mean():.3g} | "
            f"{np.percentile(sweep_err[n_pan], 95):.3g} | {1e3 * sweep_cost[n_pan]:.3g} ms |"
            for n_pan in PANEL_SWEEP
        ],
        "",
        f"The coarsest panel method at least as accurate as the PINN's pressure lift (mean "
        f"{acc['pinn_pressure'].mean():.3g}) has **{n_equal} panels**: "
        f"{1e3 * sweep_cost[n_equal]:.3g} ms per case, against the PINN's "
        f"{1e6 * per_gpu:.3g} µs (pressure, GPU, contours included) and {1e6 * per_circ:.3g} µs "
        f"(circulation, GPU) at n = {big}: {sweep_cost[n_equal] / per_gpu:.3g}× and "
        f"{sweep_cost[n_equal] / per_circ:.3g}× faster per case. **Break-even against it**: "
        f"{n_star_eq:,.0f} evaluations (pressure) or {n_star_eq_circ:,.0f} (circulation), "
        "training time charged, search not.",
        "",
        f"## Design loop (a): the incidence for cl = {TARGET_CL}",
        "",
        "Panel: Brent's method on [−10°, 15°], tolerance 1e-6 rad, one section at a time "
        f"({sum(r[3] for r in inverse_rows):.3g} s for all {len(inv)}). PINN, Newton: all "
        f"{len(inv)} sections at once from α = 0, d cl/dα by autograd, {newton_evals} batched "
        f"network evaluations, {t_newton:.3g} s. PINN, sweep: the whole cl–α curve of every "
        f"section ({len(alphas)} incidences, {t_pinn:.3g} s), then interpolation. The PINN's "
        "answer is checked by a panel solve at its incidence.",
        "",
        "| Section | α panel [deg] | Panel solves | Panel time [s] | α PINN, Newton [deg] "
        "| α PINN, sweep [deg] | cl (panel) at α PINN |",
        "|---|---|---|---|---|---|---|",
    ]
    for code, a_panel, calls, t_panel, a_pinn, a_sweep, cl_true in inverse_rows:
        lines.append(
            f"| NACA {code} | {a_panel / DEG:.3f} | {calls} | {t_panel:.3g} | {a_pinn / DEG:.3f} "
            f"| {a_sweep / DEG:.3f} | {cl_true:.3f} |"
        )
    carpet_err = np.abs(carpet_pinn - carpet_panel)
    lines += [
        "",
        "## Design loop (b): a cl carpet over camber and thickness",
        "",
        f"{g} × {g} sections (m 0–6 %, t 8–18 %, p = 0.4, α = 4°): panel {t_carpet_panel:.3g} s, "
        f"PINN {t_carpet_pinn:.3g} s ({t_carpet_panel / t_carpet_pinn:.3g}× faster); "
        f"cl difference mean {carpet_err.mean():.3g}, max {carpet_err.max():.3g}.",
    ]
    (RESULTS / "surrogate_benchmark.md").write_text("\n".join(lines) + "\n")
    for args_ in [
        (
            "inverse_alpha",
            float(np.mean([abs(r[6] - TARGET_CL) for r in inverse_rows])),
            t_newton,
            n_par,
            gpu,
            "batched Newton on alpha, 6 sections; error = |cl_panel(alpha_PINN) - 0.8|",
            {
                "panel_time_s": float(sum(r[3] for r in inverse_rows)),
                "newton_evaluations": newton_evals,
                "sweep_time_s": t_pinn,
                "alpha_err_deg_mean": float(
                    np.mean([abs(r[4] - r[1]) / DEG for r in inverse_rows])
                ),
            },
        ),
        (
            f"carpet_{g}x{g}",
            float(carpet_err.mean()),
            t_carpet_pinn,
            n_par,
            gpu,
            f"cl carpet, {g * g} sections",
            {"panel_time_s": t_carpet_panel},
        ),
    ]:
        if record:
            budget_row(*args_)
    for n_pan in PANEL_SWEEP if record else ():
        budget_row(
            f"panel_N{n_pan}",
            float(sweep_err[n_pan].mean()),
            sweep_cost[n_pan],
            n_pan,
            "cpu",
            f"panel-count sweep: one lift evaluation (time per case over {len(q_cost)} cases)",
            {
                "cl_err_p95": float(np.percentile(sweep_err[n_pan], 95)),
                "equal_accuracy": n_pan == n_equal,
            },
        )

    # ---- figures ---------------------------------------------------------------------------
    plotting.use_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = np.array(sizes)
    series = [
        ("panel", f"panel method, {PANELS} panels (CPU)", plotting.REFERENCE, "--"),
        ("pinn_gpu_pressure", "PINN, GPU", plotting.SERIES[0], "-"),
        ("pinn_cpu", "PINN, CPU", plotting.SERIES[1], "-"),
    ]
    for key, lab, col, ls in series:
        ax.loglog(ns, [timing[key][n] for n in sizes], ls, color=col, marker="o", label=lab)
    ax.loglog(
        ns,
        ns * sweep_cost[n_equal],
        "--",
        color=plotting.TEXT_MUTED,
        marker="o",
        label=f"panel method, {n_equal} panels (as accurate as the PINN)",
    )
    ax.loglog(
        ns,
        [train_s + timing["pinn_gpu_pressure"][n] for n in sizes],
        ":",
        color=plotting.SERIES[0],
        label="PINN, GPU, + training",
    )
    ax.set_xlabel("Lift evaluations n")
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Cost of n lift evaluations")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=8)
    plotting.save(fig, "surrogate_timing.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, z, title in (
        (axes[0], carpet_panel, "panel method"),
        (axes[1], carpet_pinn, "PINN"),
    ):
        cs = ax.contour(
            mm * 100, tt * 100, z.reshape(mm.shape), levels=np.arange(0.3, 1.3, 0.05), cmap="Blues"
        )
        ax.clabel(cs, fontsize=7, fmt="%.2f")
        ax.set_title(f"cl at α = 4°, p = 0.4: {title}")
        ax.set_xlabel("max camber m [% c]")
    axes[0].set_ylabel("thickness t [% c]")
    plotting.save(fig, "surrogate_carpet.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(
        [sweep_cost[n] for n in PANEL_SWEEP],
        [sweep_err[n].mean() for n in PANEL_SWEEP],
        "--",
        color=plotting.REFERENCE,
        marker="o",
        label="panel method (CPU), 8–200 panels",
    )
    for n_pan in (8, 16, 64, 200):
        ax.annotate(
            f"{n_pan}",
            (sweep_cost[n_pan], sweep_err[n_pan].mean()),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    for key, per, lab, col in (
        ("pinn_pressure", per_gpu, "PINN, pressure, GPU", plotting.SERIES[0]),
        ("pinn_pressure", per_cpu, "PINN, pressure, CPU", plotting.SERIES[1]),
        ("pinn_circulation", per_circ, "PINN, circulation, GPU", plotting.SERIES[2]),
    ):
        ax.loglog(per, acc[key].mean(), "s", color=col, ms=8, label=lab)
    ax.set_xlabel(f"Time per lift evaluation [s] (PINN: batches of {big})")
    ax.set_ylabel("Mean abs cl error vs 400 panels")
    ax.set_title("Accuracy against cost")
    ax.legend(fontsize=8)
    plotting.save(fig, "surrogate_cost_accuracy.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(cl_ref, cl_p, ".", ms=3, color=plotting.SERIES[0], label="surface pressure")
    ax.plot(cl_ref, cl_c, ".", ms=3, color=plotting.SERIES[1], label="circulation")
    lim = [cl_ref.min(), cl_ref.max()]
    ax.plot(lim, lim, "--", color=plotting.REFERENCE, lw=1)
    ax.set_xlabel("cl, panel method (400 panels)")
    ax.set_ylabel("cl, PINN")
    ax.set_title(f"Lift of {n_acc} random sections")
    ax.legend()
    plotting.save(fig, "surrogate_accuracy.png")


if __name__ == "__main__":
    main()
