"""Phase 5 (exploratory): PINNs for the steady 2D Euler equations against exact solutions.

Cases:

* ``ringleb`` — smooth transonic Ringleb flow, exact Dirichlet data on the whole boundary
  (two seeds);
* ``wedge_plain`` — Mach 2 over a 10 deg wedge: the attached oblique shock, plain PINN;
* ``wedge_viscous`` — the same with artificial viscosity nu = 0.01 in the residual;
* ``wedge_weighted`` — the same with the residual down-weighted where the predicted velocity
  divergence is large (at the shock), so the smooth regions are not sacrificed to it.

Writes ``validation/results/euler_exploratory.md``, rows in ``compute_budget.csv``,
``phase5_histories.json``, ``checkpoints/phase5_<case>.pt`` and ``docs/figures/euler_*.png``.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinncfd import io, plotting
from pinncfd.autodiff import grad
from pinncfd.device import get_device
from pinncfd.pinn_euler_2d import EulerPINN, RinglebProblem, WedgeProblem, residual
from pinncfd.pinn_euler_2d.exact import ringleb, wedge_field
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

PHASE = "phase5"
MLP_CFG = dict(width=64, depth=5, activation="tanh")
CONFIG = dict(
    adam_steps=6000,
    lr=1e-3,
    lr_final=5e-5,
    lbfgs_iters=2000,
    eval_every=100,
    resample_every=500,
    plateau_window=500,
)
QUICK = dict(adam_steps=60, lbfgs_iters=20, eval_every=20, resample_every=20)
CASES = {
    "ringleb": dict(kind="ringleb", nu=0.0),
    "wedge_plain": dict(kind="wedge", nu=0.0),
    "wedge_viscous": dict(kind="wedge", nu=0.01),
    "wedge_weighted": dict(kind="wedge", nu=0.0, weighted=True),
}
LABELS = {
    "ringleb": "Ringleb flow (smooth, transonic)",
    "wedge_plain": "wedge, plain PINN",
    "wedge_viscous": "wedge, artificial viscosity ν = 0.01",
    "wedge_weighted": "wedge, residual down-weighted at the shock",
}
SEEDS = {"ringleb": (0, 1), "wedge_plain": (0,), "wedge_viscous": (0,), "wedge_weighted": (0,)}


class WeightedWedge(WedgeProblem):
    """Residual weighted by 1 / (1 + |div v|): large velocity divergence marks the shock (it is
    zero in both uniform regions). The weight is detached, so it steers but is not optimised."""

    def losses(self, model):
        out = super().losses(model)
        w = model(self.xy_in)
        div = grad(w[:, 1:2], self.xy_in)[:, 0:1] + grad(w[:, 2:3], self.xy_in)[:, 1:2]
        lam = 1.0 / (1.0 + div.abs().detach())
        out["pde"] = (lam * residual(model, self.xy_in, self.nu).pow(2)).mean()
        return out


def make(case, device, seed, quick):
    spec = CASES[case]
    n = dict(n_interior=256, n_boundary=64) if quick else {}
    if spec["kind"] == "ringleb":
        model = EulerPINN(
            dict(MLP_CFG, seed=seed), center=(-0.1, 0.0), scale=(1.3, 2.3), ref=[0.8, 0.8, 0.8, 0.5]
        )
        problem = RinglebProblem(model.to(device), nu=spec["nu"], device=device, seed=seed, **n)
    else:
        model = EulerPINN(
            dict(MLP_CFG, seed=seed), center=(0.5, 0.5), scale=(0.5, 0.5), ref=[1.0, 2.0, 0.3, 0.7]
        )
        cls = WeightedWedge if spec.get("weighted") else WedgeProblem
        problem = cls(model.to(device), nu=spec["nu"], device=device, seed=seed, **n)
    return model, problem


def write_table(rows_by_case):
    c = CONFIG
    lines = [
        "# Phase 5 — Exploratory: PINNs for the steady 2D Euler equations",
        "",
        "Conservation-form residuals, primitive outputs (density and pressure through a "
        f"softplus), {MLP_CFG['depth']} × {MLP_CFG['width']} tanh network, 8192 interior "
        f"points. Training: {c['adam_steps']} Adam steps (lr {c['lr']:g} → {c['lr_final']:g}), "
        f"then up to {c['lbfgs_iters']} L-BFGS iterations.",
        "",
        "State error: relative L2 norm of (ρ, u, v, p) against the exact solution (Ringleb: "
        "4096 area-uniform points; wedge: 101 × 101 grid). Wedge: 'smooth' excludes a band of "
        "±0.05 around the exact shock; the shock angle is where the predicted density crosses "
        "its mid value on x = 0.4, 0.6, 0.8, 1.0 (exact β = 39.31°).",
        "",
        "| Case | Seeds | State error | ρ error | p error | Smooth-region error | β [deg] "
        "| Time [s] | Steps to 1e-1 | Steps to 3e-2 | Steps to 1e-2 | Plateau step |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case, rs in rows_by_case.items():
        m = [json.loads(r["metrics_json"]) for r in rs]
        wedge = "beta_deg" in m[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABELS[case],
                    str(len(rs)),
                    mean_std([float(r["error"]) for r in rs]),
                    mean_std([x["err_rho"] for x in m]),
                    mean_std([x["err_p"] for x in m]),
                    mean_std([x["error_smooth"] for x in m]) if wedge else "—",
                    mean_std([x["beta_deg"] for x in m]) if wedge else "—",
                    mean_std([float(r["time_total_s"]) for r in rs]),
                    fmt_steps([x["steps_to_1e-01"] for x in m]),
                    fmt_steps([x["steps_to_3e-02"] for x in m]),
                    fmt_steps([x["steps_to_1e-02"] for x in m]),
                    fmt_steps([r["plateau_step"] for r in rs]),
                ]
            )
            + " |"
        )
    dev = next(iter(rows_by_case.values()))[0]["device"]
    lines += ["", f"Device: {dev} (clock capped at 1000 MHz). Training times exclude evaluation."]
    (RESULTS / "euler_exploratory.md").write_text("\n".join(lines) + "\n")


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
    ax.set_ylabel("Relative L2 state error")
    ax.set_title("Euler PINNs: convergence (seed 0)")
    ax.legend(fontsize=8)
    plotting.save(fig, "euler_convergence.png")

    # wedge density fields
    g = np.linspace(0, 1, 201)
    xx, yy = np.meshgrid(g, g)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    below = pts[:, 1] < pts[:, 0] * np.tan(np.radians(10))
    panels = [("exact", None)] + [
        (c, c) for c in ("wedge_plain", "wedge_viscous", "wedge_weighted")
    ]
    short = {
        "exact": "exact",
        "wedge_plain": "plain PINN",
        "wedge_viscous": "artificial viscosity ν = 0.01",
        "wedge_weighted": "down-weighted at the shock",
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), sharey=True)
    for ax, (title, case) in zip(axes, panels, strict=True):
        if case is None:
            rho = wedge_field(*pts.T, np.radians(10), 2.0)[:, 0]
        else:
            model, _ = io.load_checkpoint(CHECKPOINTS / f"phase5_{case}.pt", EulerPINN, device)
            with torch.no_grad():
                rho = model(torch.tensor(pts, dtype=torch.float32, device=device))[:, 0]
            rho = rho.cpu().numpy()
        rho = np.where(below, np.nan, rho)
        im = ax.pcolormesh(
            xx, yy, rho.reshape(xx.shape), cmap="Blues", vmin=0.9, vmax=1.6, shading="auto"
        )
        ax.plot(
            [0, 1 / np.tan(np.radians(39.31))], [0, 1], color=plotting.SERIES[1], lw=0.8, ls="--"
        )
        ax.set_title(short[title], fontsize=10)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, label="density ρ / ρ∞", shrink=0.8)
    plotting.save(fig, "euler_wedge_density.png")

    # Ringleb: Mach number, exact and PINN error
    model, _ = io.load_checkpoint(CHECKPOINTS / "phase5_ringleb.pt", EulerPINN, device)
    k = np.linspace(0.7, 1.2, 60)
    qn = np.linspace(0, 1, 120)
    K, QN = np.meshgrid(k, qn)
    Q = 0.5 + (K - 0.5) * QN * (1 - 1e-9)  # up to q = k, where the branches meet at y = 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for b in (1, -1):
        x, y, rho, u, v, p = ringleb(Q, K, b)
        mach = np.hypot(u, v) / np.sqrt(1.4 * p / rho)
        with torch.no_grad():
            w = (
                model(
                    torch.tensor(
                        np.column_stack([x.ravel(), y.ravel()]), dtype=torch.float32, device=device
                    )
                )
                .cpu()
                .numpy()
            )
        ref = np.column_stack([rho.ravel(), u.ravel(), v.ravel(), p.ravel()])
        err = np.linalg.norm(w - ref, axis=1).reshape(x.shape)
        im0 = axes[0].contourf(x, y, mach, levels=np.linspace(0.3, 1.5, 13), cmap="Blues")
        axes[0].contour(x, y, mach, levels=[1.0], colors=plotting.SERIES[1], linewidths=1)
        im1 = axes[1].contourf(x, y, np.log10(err), levels=np.linspace(-5, -1, 9), cmap="Blues")
    axes[0].set_title("Ringleb flow: exact Mach number (sonic line)")
    axes[1].set_title("PINN |Δ(ρ, u, v, p)|, log10 (seed 0)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    plotting.save(fig, "euler_ringleb.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    args = ap.parse_args()
    device = get_device()
    hist_path = (SCRATCH if args.quick else RESULTS) / "phase5_histories.json"
    if not args.plots_only:
        histories = load_histories(hist_path) if hist_path.exists() else {}
        for case in args.cases:
            for seed in SEEDS[case][:1] if args.quick else SEEDS[case]:
                seed_everything(seed)
                model, problem = make(case, device, seed, args.quick)
                cfg = TrainConfig(**{**CONFIG, **(QUICK if args.quick else {})}, seed=seed)
                result = train(model, problem, cfg, device)
                histories[f"{case}/{seed}"] = result.history
                row = record(PHASE, case, seed, model, problem, result, cfg, device, not args.quick)
                f = result.final
                print(
                    f"{case:15s} seed {seed}: error {f['error']:.3e}"
                    + (f", beta {f['beta_deg']:.2f} deg" if "beta_deg" in f else "")
                    + f", {result.time_total_s:.0f} s, plateau {row['plateau_step']} "
                    f"{result.stopped}",
                    flush=True,
                )
                if seed == 0 and not args.quick:
                    io.save_checkpoint(model, CHECKPOINTS / f"phase5_{case}.pt", {"seed": 0})
        save_histories(hist_path, histories)
        if args.quick:
            return
        budget = [r for r in io.read_budget() if r["phase"] == PHASE]
        write_table({c: [r for r in budget if r["case"] == c] for c in CASES})
    figures(load_histories(hist_path), device)


if __name__ == "__main__":
    main()
