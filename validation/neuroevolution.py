"""Phase 3: NSGA-II search over the airfoil PINN's architecture and loss weights.

1. Search: population 8, 5 generations; every genome trains with the same short proxy budget
   and is scored on the 8 validation cases (objectives: velocity error, parameter count).
   Generation 0 contains the hand-tuned Phase 2 genome. Each evaluation is a row of the
   compute budget (phase3 / search_NNN).
2. Retrain the front's most accurate genome (the champion) with the full Phase 2 budget over
   three seeds, and compare it with the Phase 2 model on the test set, which the search never
   saw.

Writes ``validation/results/neuroevolution.md``, ``phase3_search.json`` (the resumable search
state), ``phase3_histories.json``, ``checkpoints/phase3_champion_s<seed>.pt`` and
``docs/figures/neuro_*.png``. ``--quick`` runs a tiny search into ``validation/runs/``.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from pinncfd import io, plotting
from pinncfd.device import get_device
from pinncfd.neuroevolution_search import genome as G
from pinncfd.neuroevolution_search.fitness import evaluate_genome
from pinncfd.neuroevolution_search.search import pareto_front, run_evolution
from pinncfd.pinn_airfoil_flowfield import AirfoilPINN, build_reference, family
from pinncfd.pinn_airfoil_flowfield.problem import evaluate

from airfoil_flowfield import CONFIG as PHASE2_CONFIG
from airfoil_flowfield import PROBLEM_CFG, TEST_KEYS
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

PHASE = "phase3"
POPULATION, GENERATIONS = 8, 5
PROXY = dict(adam_steps=400, lbfgs_iters=100, eval_every=100, resample_every=100)
QUICK_PROXY = dict(adam_steps=30, lbfgs_iters=10, eval_every=10, resample_every=10)
SEEDS = (0, 1, 2)


def search(val_ref, device, quick):
    out = SCRATCH if quick else RESULTS
    ck = out / "phase3_search.json"
    budget = QUICK_PROXY if quick else PROXY
    pcfg = dict(PROBLEM_CFG)
    if quick:
        pcfg.update(n_cases=4, n_body=32, n_far=16, n_data_cases=4, n_data=32)
    done = len(json.loads(ck.read_text())["evaluated"]) if ck.exists() else 0
    counter = [done]

    def fitness(genome, seed):
        g = dict(genome, n_interior=64) if quick else genome
        fit, model, problem, result, cfg = evaluate_genome(g, val_ref, device, seed, budget, pcfg)
        k = counter[0]
        counter[0] += 1
        if model is not None and not quick:
            row = record(PHASE, f"search_{k:03d}", seed, model, problem, result, cfg, device)
            fit["steps_to_1e-01"] = json.loads(row["metrics_json"])["steps_to_1e-01"]
        elif not quick:
            io.record_run(
                {
                    "phase": PHASE,
                    "case": f"search_{k:03d}",
                    "seed": seed,
                    "error_name": "error",
                    "error": float("inf"),
                    "n_params": fit["n_params"],
                    "time_total_s": fit["time_s"],
                    "notes": f"diverged: {fit['failed']}",
                }
                | io.environment(device)
            )
        print(f"  [{k:03d}] error {fit['error']:.4g}  {G.describe(genome)}", flush=True)
        return fit

    gens = 2 if quick else GENERATIONS
    pop = 4 if quick else POPULATION
    return run_evolution(
        fitness,
        population_size=pop,
        n_generations=gens,
        seed=0,
        checkpoint=ck,
        initial=[G.PHASE2_GENOME],
    )


def retrain(genome, val_ref, device, quick):
    """The champion at the full Phase 2 budget, over seeds."""
    rows, histories = [], {}
    budget = dict(PHASE2_CONFIG)
    pcfg = dict(PROBLEM_CFG)
    if quick:
        budget.update(QUICK_PROXY)
        pcfg.update(n_cases=4, n_body=32, n_far=16, n_data_cases=4, n_data=32)
        genome = dict(genome, n_interior=64)
    for seed in SEEDS[:1] if quick else SEEDS:
        fit, model, problem, result, cfg = evaluate_genome(
            genome, val_ref, device, seed, budget, pcfg
        )
        row = record(PHASE, "champion", seed, model, problem, result, cfg, device, not quick)
        rows.append(row)
        histories[f"champion/{seed}"] = result.history
        print(f"champion seed {seed}: val error {fit['error']:.4g}", flush=True)
        if not quick:
            io.save_checkpoint(model, CHECKPOINTS / f"phase3_champion_s{seed}.pt", {"seed": seed})
    return rows, histories


def write_table(state, champion, rows, test):
    evaluated = state["evaluated"]
    cost = sum(e["fitness"]["time_s"] for e in evaluated)
    lines = [
        "# Phase 3 — Neuroevolutionary search over architecture and loss weights",
        "",
        f"NSGA-II (ported from the RUL project), population {POPULATION}, {GENERATIONS} "
        f"generations, {len(evaluated)} genomes trained. Objectives: validation velocity error "
        "(8 cases) and parameter count, both minimised. Generation 0 includes the hand-tuned "
        "Phase 2 genome.",
        f"Proxy budget per genome: {PROXY['adam_steps']} Adam steps + {PROXY['lbfgs_iters']} "
        f"L-BFGS iterations. Total search cost: {cost / 60:.1f} min of training on "
        f"{rows[0]['device']} (clock capped at 1000 MHz).",
        "",
        "## Search history",
        "",
        "| Generation | Best error | Median error | Parameters (min–max) | Front size "
        "| Genomes trained |",
        "|---|---|---|---|---|---|",
    ]
    for h in state["history"]:
        lines.append(
            f"| {h['generation']} | {h['best_error']:.4g} | {h['median_error']:.4g} "
            f"| {h['min_params']}–{h['max_params']} | {h['front_size']} | {h['evaluations']} |"
        )
    p2 = next(e for e in evaluated if e["genome"] == G.PHASE2_GENOME)
    lines += [
        "",
        f"Hand-tuned Phase 2 genome at the proxy budget: error {p2['fitness']['error']:.4g}, "
        f"{p2['fitness']['n_params']} parameters.",
        "",
        "## Final Pareto front",
        "",
        "| Error | Parameters | Genome |",
        "|---|---|---|",
    ]
    for ind in pareto_front(state["population"]):
        f = ind["fitness"]
        lines.append(f"| {f['error']:.4g} | {f['n_params']} | {G.describe(ind['genome'])} |")
    lines += [
        "",
        "## Champion vs hand-tuned, full Phase 2 budget",
        "",
        f"Champion: {G.describe(champion)}.",
        f"Budget: {PHASE2_CONFIG['adam_steps']} Adam steps + up to "
        f"{PHASE2_CONFIG['lbfgs_iters']} L-BFGS iterations, seeds {', '.join(map(str, SEEDS))}.",
        "",
        "| Model | Val velocity error | Test velocity error | Test Cp RMS | Test cl error "
        "(circulation) | Training time [s] | Steps to 3e-2 | Steps to 1e-2 | Plateau step |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, (rs, ts) in test.items():
        mm = [json.loads(r["metrics_json"]) for r in rs]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    mean_std([float(r["error"]) for r in rs]),
                    mean_std([t["error"] for t in ts]),
                    mean_std([t["cp_rms"] for t in ts]),
                    mean_std([t["cl_err_circulation"] for t in ts]),
                    mean_std([float(r["time_total_s"]) for r in rs]),
                    fmt_steps([x["steps_to_3e-02"] for x in mm]),
                    fmt_steps([x["steps_to_1e-02"] for x in mm]),
                    fmt_steps([r["plateau_step"] for r in rs]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Test columns: mean over the seeds whose networks were saved (all three for the "
        "champion, seed 0 only for Phase 2).",
    ]
    (RESULTS / "neuroevolution.md").write_text("\n".join(lines) + "\n")


def figures(state, histories):
    plotting.use_style()
    ev = state["evaluated"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    gens = sorted({e["generation"] for e in ev})
    cmap = plt.get_cmap("Blues")
    for g in gens:
        here = [e["fitness"] for e in ev if e["generation"] == g]
        pts = [(f["n_params"], f["error"]) for f in here]
        pts = [p for p in pts if np.isfinite(p[1])]
        if pts:
            x, y = zip(*pts, strict=True)
            ax.scatter(x, y, s=22, color=cmap(0.3 + 0.7 * g / max(gens)), label=f"gen {g}")
    front = pareto_front(state["population"])
    fx = [i["fitness"]["n_params"] for i in front]
    fy = [i["fitness"]["error"] for i in front]
    ax.plot(fx, fy, color=plotting.SERIES[1], lw=1.2, marker="o", mfc="none", label="final front")
    p2 = next(e for e in ev if e["genome"] == G.PHASE2_GENOME)
    ax.scatter(
        [p2["fitness"]["n_params"]],
        [p2["fitness"]["error"]],
        marker="*",
        s=140,
        color=plotting.REFERENCE,
        label="Phase 2 genome",
        zorder=5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Validation velocity error (proxy budget)")
    ax.set_title("Every genome trained, and the final Pareto front")
    ax.legend(fontsize=8, ncol=2)
    plotting.save(fig, "neuro_pareto.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    p2h = load_histories(RESULTS / "phase2_histories.json")
    plotting.convergence_plot(
        ax,
        p2h["physics_data/0"],
        label="Phase 2 (hand-tuned)",
        color=plotting.SERIES[0],
        adam_steps=PHASE2_CONFIG["adam_steps"],
    )
    plotting.convergence_plot(
        ax, histories["champion/0"], label="Phase 3 champion", color=plotting.SERIES[1]
    )
    ax.set_ylabel("Validation velocity error")
    ax.set_title("Full-budget training, seed 0")
    ax.legend()
    plotting.save(fig, "neuro_convergence.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    device = get_device()
    val_ref = build_reference(family.val_cases(), seed=1)
    hist_path = (SCRATCH if args.quick else RESULTS) / "phase3_histories.json"

    if not args.plots_only:
        state = search(val_ref, device, args.quick)
        champion = pareto_front(state["population"])[0]["genome"]
        print("champion:", G.describe(champion), flush=True)
        rows, histories = retrain(champion, val_ref, device, args.quick)
        save_histories(hist_path, histories)
        if args.quick:
            return
        test_ref = build_reference(family.test_cases(), seed=2)
        ts_champ = []
        for seed in SEEDS:
            model, _ = io.load_checkpoint(
                CHECKPOINTS / f"phase3_champion_s{seed}.pt", AirfoilPINN, device
            )
            ts_champ.append(evaluate(model, test_ref, device))
        model, _ = io.load_checkpoint(CHECKPOINTS / "phase2_physics_data.pt", AirfoilPINN, device)
        ts_p2 = [evaluate(model, test_ref, device)]
        p2_rows = [
            r for r in io.read_budget() if r["phase"] == "phase2" and r["case"] == "physics_data"
        ]
        for seed, t in zip(SEEDS, ts_champ, strict=True):
            r = next(
                r for r in io.read_budget() if r["case"] == "champion" and r["seed"] == str(seed)
            )
            m = json.loads(r["metrics_json"])
            m.update({f"test_{k}": t[k] for k in TEST_KEYS})
            r["metrics_json"] = json.dumps(m)
            io.record_run(r)
        rows = [r for r in io.read_budget() if r["phase"] == PHASE and r["case"] == "champion"]
        write_table(
            state,
            champion,
            rows,
            {"Phase 2 (hand-tuned)": (p2_rows, ts_p2), "Phase 3 champion": (rows, ts_champ)},
        )
    state = json.loads((RESULTS / "phase3_search.json").read_text())
    figures(state, load_histories(hist_path))


if __name__ == "__main__":
    main()
