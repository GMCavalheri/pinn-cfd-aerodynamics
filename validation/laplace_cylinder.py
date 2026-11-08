"""Phase 1 validation: PINNs for potential flow past a cylinder against the exact solution.

Trains four formulations over three seeds, times the AERO-01 panel method on the same flow,
and writes:

* ``validation/results/laplace_cylinder.md`` — results table (mean ± std over seeds);
* ``validation/results/compute_budget.csv`` — one row per run (error, time, steps);
* ``validation/results/phase1_histories.json`` — evaluated training histories;
* ``validation/results/checkpoints/phase1_<case>.pt`` — seed-0 networks;
* ``docs/figures/laplace_*.png`` — convergence, error maps, surface Cp.

``--quick`` runs tiny budgets into ``validation/runs/`` without touching the results;
``--plots-only`` redraws the figures from the saved histories and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np

from pinncfd import io, plotting
from pinncfd.device import get_device
from pinncfd.external.panel import solve_cylinder
from pinncfd.pinn_laplace_2d import CylinderPINN, CylinderProblem, exact
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

PHASE = "phase1"
GAMMA = 2 * math.pi
CASES = {
    "psi_soft": dict(formulation="psi", hard=False),
    "psi_hard": dict(formulation="psi", hard=True),
    "psi_hard_lifting": dict(formulation="psi", hard=True, gamma=GAMMA),
    "phi_soft": dict(formulation="phi", hard=False),
}
LABELS = {
    "psi_soft": "ψ, soft BCs",
    "psi_hard": "ψ, hard BCs",
    "psi_hard_lifting": "ψ, hard BCs, Γ = 2π",
    "phi_soft": "φ, soft BCs (Neumann body)",
}
MLP_CFG = dict(width=50, depth=4, activation="tanh")
N_INTERIOR = 8192
SEEDS = (0, 1, 2)
CONFIG = dict(adam_steps=5000, lr=1e-3, lr_final=1e-4, lbfgs_iters=4000, eval_every=100)
QUICK = dict(adam_steps=200, lbfgs_iters=50, eval_every=50)
PANEL_N = (64, 128, 256, 512)
R_MIN_PANEL = 1.05  # the panel near field is unresolved within a panel length of the body


def run_case(case, seed, device, quick=False):
    seed_everything(seed)
    model = CylinderPINN(**CASES[case], mlp=dict(MLP_CFG, seed=seed)).to(device)
    problem = CylinderProblem(model, n_interior=N_INTERIOR, device=device, seed=seed)
    cfg = TrainConfig(**(QUICK if quick else CONFIG), seed=seed)
    result = train(model, problem, cfg, device)
    row = record(PHASE, case, seed, model, problem, result, cfg, device, record_csv=not quick)
    return model, result, row


def panel_baseline(record_csv=True):
    """Time and score the AERO-01 panel method on the same evaluation set."""
    probe = CylinderProblem(CylinderPINN(hard=True, mlp=dict(width=4, depth=1)), device="cpu")
    xy = probe.eval_xy[np.hypot(*probe.eval_xy.T) >= R_MIN_PANEL]
    rows = []
    for gamma, name in ((0.0, "panel"), (GAMMA, "panel_lifting")):
        u_ref = np.column_stack(exact.velocity(*xy.T, gamma))
        for n in PANEL_N:
            times = []
            for _ in range(20):
                t0 = time.perf_counter()
                sol = solve_cylinder(n, gamma)
                times.append(time.perf_counter() - t0)
            u, v = sol.velocity(*xy.T)
            err = np.linalg.norm(np.column_stack([u, v]) - u_ref) / np.linalg.norm(u_ref)
            xc, yc = sol.control_points.T
            cp_err = np.abs(sol.cp - exact.surface_cp(np.arctan2(yc, xc), gamma))
            metrics = {
                "error": float(err),
                "cp_max_abs": float(cp_err.max()),
                "cp_rms": float(np.sqrt(np.mean(cp_err**2))),
                "cl": sol.cl,
                "cl_abs_err": abs(sol.cl - gamma),
            }
            row = {
                "phase": PHASE,
                "case": f"{name}_N{n}",
                "seed": 0,
                "error_name": "error",
                "error": float(err),
                "time_total_s": float(np.median(times)),
                "n_params": n,  # unknowns: one source strength per panel
                "metrics_json": json.dumps(metrics),
                "notes": f"direct solve, median of 20; velocity error on r >= {R_MIN_PANEL}",
            }
            row.update(io.environment("cpu"))
            if record_csv:
                io.record_run(row)
            rows.append((gamma, n, metrics, float(np.median(times))))
    return rows


def pinn_error_off_body(device="cpu"):
    """Seed-0 PINN velocity error on the panel comparison subset (r >= R_MIN_PANEL)."""
    out = {}
    for case in CASES:
        model, _ = io.load_checkpoint(CHECKPOINTS / f"phase1_{case}.pt", CylinderPINN, device)
        prob = CylinderProblem(model, n_interior=16, device=device)
        keep = np.hypot(*prob.eval_xy.T) >= R_MIN_PANEL
        _, u, v = prob.predict(model, prob.eval_xy[keep])
        ref = prob.u_ref[keep]
        out[case] = float(np.linalg.norm(np.column_stack([u, v]) - ref) / np.linalg.norm(ref))
    return out


def write_table(rows_by_case, panel_rows):
    lines = [
        "# Phase 1 — PINN for potential flow past a cylinder",
        "",
        "Unit cylinder, freestream 1, domain 1 ≤ r ≤ 5 with exact Dirichlet data on r = 5.",
        f"Network: {MLP_CFG['depth']} × {MLP_CFG['width']} tanh MLP; {N_INTERIOR} interior "
        "collocation points (redrawn every 500 Adam steps).",
        f"Training: {CONFIG['adam_steps']} Adam steps (lr {CONFIG['lr']:g} → "
        f"{CONFIG['lr_final']:g}), then up to {CONFIG['lbfgs_iters']} L-BFGS iterations.",
        f"Mean ± standard deviation over seeds {', '.join(map(str, SEEDS))}.",
        "",
        "Velocity error: relative L2 norm of (u, v) on a 64 × 128 polar grid (geometric in r).",
        "Steps: global count (Adam steps, then L-BFGS iterations after step "
        f"{CONFIG['adam_steps']}). Plateau: first step after which the best error improves by "
        "less than 1 % over any 500-step window.",
        "",
        "| Case | Velocity error | ψ/φ error | max abs ΔCp | cl error | Adam time [s] "
        "| L-BFGS time [s] | Steps to 1e-2 | Steps to 1e-3 | Plateau step |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case, rows in rows_by_case.items():
        m = [json.loads(r["metrics_json"]) for r in rows]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABELS[case],
                    mean_std([float(r["error"]) for r in rows]),
                    mean_std([x["err_f"] for x in m]),
                    mean_std([x["cp_max_abs"] for x in m]),
                    mean_std([x["cl_abs_err"] for x in m]),
                    mean_std([float(r["time_adam_s"]) for r in rows]),
                    mean_std([float(r["time_lbfgs_s"]) for r in rows]),
                    fmt_steps([x["steps_to_1e-02"] for x in m]),
                    fmt_steps([x["steps_to_1e-03"] for x in m]),
                    fmt_steps([r["plateau_step"] for r in rows]),
                ]
            )
            + " |"
        )
    dev = rows_by_case[next(iter(rows_by_case))][0]["device"]
    lines += [
        "",
        f"Device: {dev} (clock capped at 1000 MHz). Training times exclude evaluation.",
        "",
        "## Panel-method baseline (AERO-01)",
        "",
        f"Velocity errors here use the grid points with r ≥ {R_MIN_PANEL} only: on the circle "
        "itself the panel solution is singular at the polygon nodes. On that subset the seed-0 "
        "PINNs score "
        + ", ".join(f"{LABELS[c]} {e:.3g}" for c, e in pinn_error_off_body().items())
        + ".",
        "",
        "| Flow | Panels | Velocity error | max abs ΔCp (control points) | Solve time [ms] |",
        "|---|---|---|---|---|",
    ]
    for gamma, n, m, t in panel_rows:
        flow = "non-lifting" if gamma == 0 else "Γ = 2π (imposed)"
        lines.append(f"| {flow} | {n} | {m['error']:.3g} | {m['cp_max_abs']:.3g} | {1e3 * t:.3g} |")
    (RESULTS / "laplace_cylinder.md").write_text("\n".join(lines) + "\n")


def figures(histories, device="cpu"):
    plotting.use_style()
    # convergence
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, case in enumerate(CASES):
        h = histories[f"{case}/0"]
        plotting.convergence_plot(
            ax,
            h,
            label=LABELS[case],
            color=plotting.SERIES[i],
            adam_steps=CONFIG["adam_steps"] if i == 0 else None,
        )
    ax.axhline(1e-2, color=plotting.TEXT_MUTED, lw=0.8, ls="--")
    ax.set_ylabel("Relative L2 velocity error")
    ax.set_title("Training convergence, cylinder (seed 0)")
    ax.legend()
    plotting.save(fig, "laplace_convergence.png")

    # loss terms of the soft formulation: the body term is the one that lags
    fig, ax = plt.subplots(figsize=(7, 4))
    h = histories["psi_soft/0"]
    steps = [r["step"] for r in h]
    for i, k in enumerate(("loss_pde", "loss_body", "loss_far")):
        ax.semilogy(steps, [r[k] for r in h], color=plotting.SERIES[i], label=k[5:])
    ax.axvline(CONFIG["adam_steps"], color=plotting.TEXT_MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Step (Adam, then L-BFGS iterations)")
    ax.set_ylabel("Mean-square loss term")
    ax.set_title("Soft constraints: loss terms (ψ, seed 0)")
    ax.legend()
    plotting.save(fig, "laplace_soft_loss_terms.png")

    # error maps and surface Cp from the seed-0 checkpoints
    x = np.linspace(-3, 3, 241)
    xx, yy = np.meshgrid(x, x)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = np.hypot(*pts.T) < 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, case in zip(axes, ("psi_soft", "psi_hard"), strict=True):
        model, _ = io.load_checkpoint(CHECKPOINTS / f"phase1_{case}.pt", CylinderPINN, device)
        prob = CylinderProblem(model, n_interior=16, device=device)
        _, u, v = prob.predict(model, pts)
        ue, ve = exact.velocity(*pts.T)
        err = np.hypot(u - ue, v - ve)
        err[inside] = np.nan
        im = ax.pcolormesh(
            xx,
            yy,
            np.log10(err.reshape(xx.shape)),
            cmap="Blues",
            vmin=-5,
            vmax=-1,
            shading="auto",
        )
        th = np.linspace(0, 2 * np.pi, 200)
        ax.fill(np.cos(th), np.sin(th), color=plotting.GRID)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(LABELS[case])
        ax.set_xlabel("x / R")
    axes[0].set_ylabel("y / R")
    fig.colorbar(im, ax=axes, label="log10 |velocity error|")
    plotting.save(fig, "laplace_error_maps.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    th = np.linspace(0, 2 * np.pi, 361)
    for i, case in enumerate(("psi_soft", "psi_hard_lifting")):
        model, _ = io.load_checkpoint(CHECKPOINTS / f"phase1_{case}.pt", CylinderPINN, device)
        prob = CylinderProblem(model, n_interior=16, device=device)
        _, u, v = prob.predict(model, np.column_stack([np.cos(th), np.sin(th)]))
        ax.plot(np.degrees(th), 1 - u**2 - v**2, color=plotting.SERIES[i], label=LABELS[case])
        ax.plot(
            np.degrees(th),
            exact.surface_cp(th, model.gamma),
            color=plotting.REFERENCE,
            ls="--",
            lw=1.0,
            label="exact" if i == 0 else None,
        )
    ax.set_xlabel("θ [deg]")
    ax.set_ylabel("Pressure coefficient Cp")
    ax.set_title("Surface pressure, PINN vs exact (seed 0)")
    ax.legend()
    plotting.save(fig, "laplace_surface_cp.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--tables-only", action="store_true", help="redo panel baseline and table")
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    args = ap.parse_args()
    device = get_device()
    hist_path = (SCRATCH if args.quick else RESULTS) / "phase1_histories.json"

    if not args.plots_only:
        histories = load_histories(hist_path) if hist_path.exists() else {}
        for case in [] if args.tables_only else args.cases:
            for seed in SEEDS[:1] if args.quick else SEEDS:
                model, result, row = run_case(case, seed, device, args.quick)
                histories[f"{case}/{seed}"] = result.history
                print(
                    f"{case:18s} seed {seed}: error {float(row['error']):.3e}, "
                    f"adam {result.time_adam_s:.0f} s, lbfgs {result.time_lbfgs_s:.0f} s "
                    f"({result.lbfgs_iters} it), to 1e-2: {row['steps_to_target']}, "
                    f"plateau: {row['plateau_step']} {result.stopped}",
                    flush=True,
                )
                if seed == 0 and not args.quick:
                    io.save_checkpoint(model, CHECKPOINTS / f"phase1_{case}.pt", {"seed": 0})
        save_histories(hist_path, histories)
        if args.quick:
            return
        panel_rows = panel_baseline()
        budget = [r for r in io.read_budget() if r["phase"] == PHASE]
        by_case = {c: [r for r in budget if r["case"] == c] for c in CASES}
        write_table(by_case, panel_rows)
    figures(load_histories(hist_path))


if __name__ == "__main__":
    main()
