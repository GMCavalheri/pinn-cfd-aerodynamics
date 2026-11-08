"""Phase 2 validation: a PINN for the NACA 4-digit family against the AERO-01 panel method.

Runs the main model (physics + panel data) over seeds plus two single-seed ablations
(physics only, data only), then scores the seed-0 networks on the held-out test sections, on
extrapolation cases outside the family, and times the panel method on the same cases.

Writes ``validation/results/airfoil_flowfield.md``, rows in ``compute_budget.csv``,
``phase2_histories.json``, ``checkpoints/phase2_<case>.pt`` and ``docs/figures/airfoil_*.png``.
``--quick`` runs tiny budgets into ``validation/runs/``; ``--plots-only`` redraws.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np

from pinncfd import io, plotting
from pinncfd.device import get_device
from pinncfd.external.geometry import Joukowski, contour, inside, naca, panel_solution, solve_panels
from pinncfd.pinn_airfoil_flowfield import AirfoilPINN, AirfoilProblem, build_reference, family
from pinncfd.pinn_airfoil_flowfield.problem import evaluate, lift_from_circulation, predict_velocity
from pinncfd.seed import seed_everything
from pinncfd.training import TrainConfig, train

from common import (
    CHECKPOINTS,
    RESULTS,
    SCRATCH,
    fmt_steps,
    load_histories,
    mean_std,
    record,
    save_histories,
)

PHASE = "phase2"
DEG = np.pi / 180
CASES = {
    "physics_data": dict(use_physics=True, use_data=True),
    "physics_only": dict(use_physics=True, use_data=False),
    "data_only": dict(use_physics=False, use_data=True),
    "plain_features": dict(use_physics=True, use_data=True),
}
MODEL_KW = {"plain_features": dict(features="compress")}
LABELS = {
    "physics_data": "physics + panel data",
    "physics_only": "physics only",
    "data_only": "panel data only",
    "plain_features": "physics + data, plain coordinates",
}
SEEDS = {"physics_data": (0, 1, 2), "physics_only": (0,), "data_only": (0,), "plain_features": (0,)}
MLP_CFG = dict(width=64, depth=5, activation="tanh")
HEAD_CFG = dict(width=32, depth=2, activation="tanh")
PROBLEM_CFG = dict(n_cases=32, n_interior=384, n_body=128, n_far=64, n_data_cases=16, n_data=128)
CONFIG = dict(
    adam_steps=4000,
    lr=1e-3,
    lr_final=5e-5,
    lbfgs_iters=1000,
    eval_every=250,
    resample_every=250,
    plateau_window=1000,
)
QUICK = dict(adam_steps=100, lbfgs_iters=20, eval_every=50, resample_every=50)
PANEL_N = 200  # the baseline resolution; the reference uses 400
TEST_KEYS = ("error", "cp_rms", "cp_rms_mid", "cp_max_abs", "cl_err_circulation", "cl_err_pressure")


def make_problem(case, model, val_ref, device, seed, quick=False):
    cfg = dict(PROBLEM_CFG)
    if quick:
        cfg.update(n_cases=4, n_interior=64, n_body=32, n_far=16, n_data_cases=4, n_data=32)
    return AirfoilProblem(model, evaluation=val_ref, device=device, seed=seed, **cfg, **CASES[case])


def run_case(case, seed, val_ref, device, quick=False):
    seed_everything(seed)
    model = AirfoilPINN(
        mlp=dict(MLP_CFG, seed=seed), head=dict(HEAD_CFG, seed=seed), **MODEL_KW.get(case, {})
    ).to(device)
    problem = make_problem(case, model, val_ref, device, seed, quick)
    cfg = TrainConfig(**{**CONFIG, **(QUICK if quick else {})}, seed=seed)
    result = train(model, problem, cfg, device)
    return model, problem, result, cfg


def panel_timing(cases, n=PANEL_N, repeats=5):
    """Median wall time of one panel solution per case: contour + assemble + solve (NumPy)."""
    times = []
    for c in cases:
        for _ in range(repeats):
            t0 = time.perf_counter()
            solve_panels(naca(c.m, c.p, c.t).coordinates(n), c.alpha)
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def panel_credibility(test_ref):
    """How good is the reference itself? N = 200 and 400 against N = 800 on the test cases,
    and N = 400 against the exact Joukowski solution."""
    out = {}
    for n in (200, 400):
        errs, cle = [], []
        for i, c in enumerate(test_ref.cases):
            hi = panel_solution(c.m, c.p, c.t, c.alpha, 800)
            lo = panel_solution(c.m, c.p, c.t, c.alpha, n)
            pts = test_ref.field_xy[i * test_ref.n_field : (i + 1) * test_ref.n_field]
            uh, vh = hi.velocity(*pts.T)
            ul, vl = lo.velocity(*pts.T)
            errs.append(np.linalg.norm([ul - uh, vl - vh]) / np.linalg.norm([uh, vh]))
            cle.append(abs(lo.cl - hi.cl))
        out[n] = (float(np.mean(errs)), float(np.mean(cle)))
    jk = Joukowski(mx=0.1, my=0.05)
    nodes = jk.coordinates(400)
    sol = solve_panels(nodes / jk.chord, 4 * DEG)
    out["joukowski_cl_rel"] = abs(sol.cl - jk.cl(4 * DEG)) / jk.cl(4 * DEG)
    return out


def write_table(rows, test_scores, extrap_scores, panel_info, cred):
    c = CONFIG
    lines = [
        "# Phase 2 — PINN for the flow around a NACA 4-digit family",
        "",
        "Family: m ∈ [0, 0.06], p ∈ [0.3, 0.5], t ∈ [0.08, 0.18], α ∈ [−4°, 8°]. One network",
        "ψ(x, y; m, p, t, α) for the whole family: freestream + vortex + learned disturbance, the",
        "network seeing inverse-Joukowski (circle-plane) coordinates; see",
        "`src/pinncfd/pinn_airfoil_flowfield/model.py`. Lift: circulation of the predicted",
        "velocity on r = 1.5 c (Kutta–Joukowski) and surface-pressure integration.",
        "Reference: AERO-01 Hess-Smith panel method with 400 panels.",
        "",
        f"Field network {MLP_CFG['depth']} × {MLP_CFG['width']} tanh, case head "
        f"{HEAD_CFG['depth']} × {HEAD_CFG['width']}. Per Adam step: "
        f"{PROBLEM_CFG['n_cases']} random shapes × {PROBLEM_CFG['n_interior']} interior points "
        f"(+ surface, Kutta, far-field points), panel data from {PROBLEM_CFG['n_data_cases']} of "
        f"the 40 training cases × {PROBLEM_CFG['n_data']} points; redrawn every "
        f"{c['resample_every']} steps.",
        f"Training: {c['adam_steps']} Adam steps (lr {c['lr']:g} → {c['lr_final']:g}), then up to "
        f"{c['lbfgs_iters']} L-BFGS iterations on the last batch.",
        "",
        "Velocity error: relative L2 norm of (u, v) at 512 field points per case (no closer "
        "than 0.02 c to the surface), averaged over cases. Validation: 8 cases (used during "
        "training to monitor convergence). Test: 6 sections × 4 angles, never used before the "
        "final evaluation. Plateau window 1000 steps, tolerance 1 %; a plateau at the last step "
        "(5000) means the run was still improving when the budget ran out.",
        "",
        "## Training runs (validation set)",
        "",
        "| Run | Seeds | Val velocity error | Val Cp RMS | Val cl error | Adam time [s] "
        "| L-BFGS time [s] | Steps to 1e-1 | Steps to 3e-2 | Steps to 1e-2 | Plateau step |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case, rs in rows.items():
        m = [json.loads(r["metrics_json"]) for r in rs]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABELS[case],
                    str(len(rs)),
                    mean_std([float(r["error"]) for r in rs]),
                    mean_std([x["cp_rms"] for x in m]),
                    mean_std([x["cl_err_circulation"] for x in m]),
                    mean_std([float(r["time_adam_s"]) for r in rs]),
                    mean_std([float(r["time_lbfgs_s"]) for r in rs]),
                    fmt_steps([x["steps_to_1e-01"] for x in m]),
                    fmt_steps([x["steps_to_3e-02"] for x in m]),
                    fmt_steps([x["steps_to_1e-02"] for x in m]),
                    fmt_steps([r["plateau_step"] for r in rs]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        f"Device: {rows['physics_data'][0]['device']} (clock capped at 1000 MHz).",
        "",
        "## Held-out test sections (seed-0 networks)",
        "",
        "| Run | Velocity error | Cp RMS | Cp RMS, 0.05 < x < 0.95 | max abs ΔCp "
        "| cl error (circulation) | cl error (pressure) |",
        "|---|---|---|---|---|---|---|",
    ]
    for case, s in test_scores.items():
        lines.append(
            f"| {LABELS[case]} | {np.mean(s['error']):.3g} | {np.mean(s['cp_rms']):.3g} "
            f"| {np.mean(s['cp_rms_mid']):.3g} | {np.mean(s['cp_max_abs']):.3g} "
            f"| {np.mean(s['cl_err_circulation']):.3g} | {np.mean(s['cl_err_pressure']):.3g} |"
        )
    s = test_scores["physics_data"]
    lines += [
        "",
        "Per test case, physics + panel data (seed 0):",
        "",
        "| Case | Velocity error | Cp RMS | cl panel | cl PINN (circulation) "
        "| cl PINN (pressure) |",
        "|---|---|---|---|---|---|",
    ]
    for i, case in enumerate(family.test_cases()):
        lines.append(
            f"| {case.name} | {s['error'][i]:.3g} | {s['cp_rms'][i]:.3g} | {s['cl_ref'][i]:.4f} "
            f"| {s['cl_circulation'][i]:.4f} | {s['cl_pressure'][i]:.4f} |"
        )
    lines += [
        "",
        "## Outside the family (seed 0, physics + panel data)",
        "",
        "| Case | Leaves the box in | Velocity error | Cp RMS | cl panel | cl PINN (circulation) |",
        "|---|---|---|---|---|---|",
    ]
    names = [["m", "p", "t", "α"][k] for k in range(4)]
    for i, case in enumerate(family.extrapolation_cases()):
        out = [
            names[k]
            for k in range(4)
            if not family.LOW[k] - 1e-12 <= case.params[k] <= family.HIGH[k] + 1e-12
        ]
        e = extrap_scores
        lines.append(
            f"| {case.name} | {', '.join(out)} | {e['error'][i]:.3g} | {e['cp_rms'][i]:.3g} "
            f"| {e['cl_ref'][i]:.4f} | {e['cl_circulation'][i]:.4f} |"
        )
    lines += [
        "",
        "## The reference and the baseline",
        "",
        f"Panel method with {PANEL_N} panels (the baseline): {1e3 * panel_info:.3g} ms per case "
        "(contour + assemble + solve, median, NumPy with its default BLAS threads).",
        "",
        "| Panels | Velocity difference to N = 800 | cl difference to N = 800 |",
        "|---|---|---|",
    ]
    for n in (200, 400):
        lines.append(f"| {n} | {cred[n][0]:.3g} | {cred[n][1]:.3g} |")
    lines += [
        "",
        f"Joukowski airfoil (mx = 0.1, my = 0.05, α = 4°), 400 panels: cl within "
        f"{100 * cred['joukowski_cl_rel']:.2f} % of the exact value.",
    ]
    (RESULTS / "airfoil_flowfield.md").write_text("\n".join(lines) + "\n")


def figures(histories, device):
    plotting.use_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, case in enumerate(CASES):
        plotting.convergence_plot(
            ax,
            histories[f"{case}/0"],
            label=LABELS[case],
            color=plotting.SERIES[i],
            adam_steps=CONFIG["adam_steps"] if i == 0 else None,
        )
    ax.set_ylabel("Validation velocity error")
    ax.set_title("Training convergence, airfoil family (seed 0)")
    ax.legend()
    plotting.save(fig, "airfoil_convergence.png")

    model, _ = io.load_checkpoint(CHECKPOINTS / "phase2_physics_data.pt", AirfoilPINN, device)
    show = [family.Case(0.0, 0.4, 0.12, 4 * DEG), family.Case(0.04, 0.4, 0.15, 4 * DEG)]
    show += [family.Case(0.05, 0.5, 0.18, 7 * DEG)]
    ref = build_reference(show, n_field=16)
    s = evaluate(model, ref, device, per_case=True)
    cp = s["cp"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for i, (ax, c) in enumerate(zip(axes, show, strict=True)):
        x = ref.surf_xy[i * ref.n_surf : (i + 1) * ref.n_surf, 0]
        plotting.plot_cp(ax, x, cp[i], plotting.SERIES[0], label="PINN")
        plotting.plot_cp(
            ax,
            x,
            ref.surf_cp.reshape(len(show), -1)[i],
            plotting.REFERENCE,
            label="panel (400)",
            ls="--",
            lw=1.0,
        )
        cl_ref, cl_pinn = s["cl_ref"][i], s["cl_circulation"][i]
        ax.set_title(f"{c.name}\ncl {cl_ref:.3f} panel, {cl_pinn:.3f} PINN", fontsize=10)
        ax.set_xlabel("x / c")
    axes[0].legend()
    plotting.save(fig, "airfoil_cp_test.png")

    # error map around one test section
    c = family.Case(0.04, 0.4, 0.15, 4 * DEG)
    sol = panel_solution(c.m, c.p, c.t, c.alpha, 400)
    x = np.linspace(-0.3, 1.3, 321)
    y = np.linspace(-0.4, 0.4, 161)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask = inside(c.m, c.p, c.t, pts)
    u, v = predict_velocity(model, pts, np.repeat(c.params[None], len(pts), 0), device)
    ur, vr = sol.velocity(*pts.T)
    err = np.hypot(u - ur, v - vr)
    err[mask] = np.nan
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.pcolormesh(
        xx, yy, np.log10(err.reshape(xx.shape)), cmap="Blues", vmin=-4, vmax=0, shading="auto"
    )
    plotting.plot_body(ax, contour(c.m, c.p, c.t, 400))
    ax.grid(False)
    ax.set_title(f"|velocity error| vs panel method, {c.name}")
    fig.colorbar(im, ax=ax, label="log10 |Δ(u, v)|")
    plotting.save(fig, "airfoil_error_map.png")

    # lift curves
    fig, ax = plt.subplots(figsize=(7, 4))
    alphas = np.linspace(-4, 8, 25) * DEG
    for i, code in enumerate(("0012", "2412", "4415")):
        # family._naca gives a symmetric section the box-centre p, as in the test set
        q = np.array([family._naca(code, a / DEG).params for a in alphas])
        cl_pinn = lift_from_circulation(model, q, device)
        ax.plot(alphas / DEG, cl_pinn, color=plotting.SERIES[i], label=f"NACA {code} PINN")
        cl = [panel_solution(*q[k, :3], q[k, 3], 200).cl for k in range(0, len(q), 3)]
        ax.plot(alphas[::3] / DEG, cl, "o", color=plotting.SERIES[i], ms=4, mfc="none")
    ax.plot([], [], "o", color=plotting.TEXT_MUTED, mfc="none", label="panel method")
    ax.set_xlabel("α [deg]")
    ax.set_ylabel("Lift coefficient cl")
    ax.set_title("Lift curves (PINN: circulation at r = 1.5 c), seed 0")
    ax.legend()
    plotting.save(fig, "airfoil_lift_curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    args = ap.parse_args()
    device = get_device()
    hist_path = (SCRATCH if args.quick else RESULTS) / "phase2_histories.json"
    val_ref = build_reference(family.val_cases(), seed=1)

    if not args.plots_only:
        histories = load_histories(hist_path) if hist_path.exists() else {}
        for case in args.cases:
            for seed in SEEDS[case][:1] if args.quick else SEEDS[case]:
                model, problem, result, cfg = run_case(case, seed, val_ref, device, args.quick)
                histories[f"{case}/{seed}"] = result.history
                row = record(PHASE, case, seed, model, problem, result, cfg, device, not args.quick)
                print(
                    f"{case:13s} seed {seed}: val error {float(row['error']):.3e}, "
                    f"adam {result.time_adam_s:.0f} s, lbfgs {result.time_lbfgs_s:.0f} s "
                    f"({result.lbfgs_iters} it), plateau {row['plateau_step']} {result.stopped}",
                    flush=True,
                )
                if seed == 0 and not args.quick:
                    io.save_checkpoint(model, CHECKPOINTS / f"phase2_{case}.pt", {"seed": 0})
        save_histories(hist_path, histories)
        if args.quick:
            return
        test_ref = build_reference(family.test_cases(), seed=2)
        extrap_ref = build_reference(family.extrapolation_cases(), seed=3)
        test_scores, extrap = {}, None
        for case in CASES:
            model, _ = io.load_checkpoint(CHECKPOINTS / f"phase2_{case}.pt", AirfoilPINN, device)
            test_scores[case] = evaluate(model, test_ref, device, per_case=True)
            if case == "physics_data":
                extrap = evaluate(model, extrap_ref, device, per_case=True)
        budget = [r for r in io.read_budget() if r["phase"] == PHASE]
        rows = {c: [r for r in budget if r["case"] == c] for c in CASES}
        # test-set metrics go into the seed-0 rows of the compute budget
        for case in CASES:
            r0 = next(r for r in rows[case] if r["seed"] == "0")
            m = json.loads(r0["metrics_json"])
            m.update({f"test_{k}": float(np.mean(test_scores[case][k])) for k in TEST_KEYS})
            r0["metrics_json"] = json.dumps(m)
            io.record_run(r0)
        write_table(
            rows,
            test_scores,
            extrap,
            panel_timing(family.test_cases()),
            panel_credibility(test_ref),
        )
    figures(load_histories(hist_path), device)


if __name__ == "__main__":
    main()
