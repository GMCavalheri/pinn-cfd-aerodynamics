"""The Phase 3 NSGA-II, run generation by generation until a compute budget is spent.

``run_evolution`` resumes from its checkpoint, so asking it for one more generation at a time
and checking the ledger in between gives a search that stops after the generation that exhausts
the budget, with every genome's cost charged, as the Optuna baselines are charged per trial.
Objectives: validation error and parameter count, as in Phase 3.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from pinncfd.benchmark import ComputeLedger
from pinncfd.neuroevolution_search.genome import GeneSpace
from pinncfd.neuroevolution_search.search import pareto_front, run_evolution


def run_nsga(
    evaluate: Callable,
    space: GeneSpace,
    ledger: ComputeLedger,
    *,
    population_size: int,
    seed: int,
    checkpoint: Path,
    initial: list[dict] | None = None,
    max_generations: int = 1000,
    log: Callable[[str], None] = print,
) -> dict:
    def fitness(genome: dict, s: int) -> dict:
        out = evaluate(genome, s, None)
        ledger.charge(out, label=f"nsga seed {s}")
        return {"error": float(out["val"]["error"]), "n_params": out["n_params"]}

    state = None
    for gen in range(max_generations):
        state = run_evolution(
            fitness,
            population_size=population_size,
            n_generations=gen + 1,
            seed=seed,
            checkpoint=checkpoint,
            initial=initial,
            log=log,
            space=space,
        )
        if ledger.exhausted:
            break
    front = pareto_front(state["population"])
    best = front[0] if front and math.isfinite(front[0]["fitness"]["error"]) else None
    return {
        "best_genome": best["genome"] if best else None,
        "best_error": best["fitness"]["error"] if best else math.inf,
        "generations": state["generation"] + 1,
        "evaluated": state["evaluated"],
        "ledger": ledger.summary(),
    }
