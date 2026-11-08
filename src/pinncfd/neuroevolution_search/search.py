"""The NSGA-II generational loop (ported from the RUL project's ``search.py``, commit addc209).

Each generation:

1. **Vary** — ``population_size`` offspring by binary tournament (better Pareto rank wins,
   ties broken by crowding distance), uniform crossover and per-gene mutation.
2. **Evaluate** — train every offspring with the proxy budget and score it.
3. **Select** — pool parents and offspring (the elitist (mu + lambda) scheme of NSGA-II, so a
   good parent is never lost to worse children) and keep the best ``population_size``.

The fitness function is passed in, so the loop knows nothing about PINNs. A JSON checkpoint is
written after every generation; rerunning resumes from it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pinncfd.neuroevolution_search.genome import crossover, mutate, sample_genome
from pinncfd.neuroevolution_search.nsga import (
    crowding_distance,
    fast_non_dominated_sort,
    select_next_generation,
)

# fitness(genome, seed) -> dict with at least "error" and "n_params" (both minimised)
Fitness = Callable[[dict, int], dict]


def objectives(population: list[dict]) -> list[tuple[float, float]]:
    out = []
    for ind in population:
        err = ind["fitness"]["error"]
        out.append((err if math.isfinite(err) else math.inf, float(ind["fitness"]["n_params"])))
    return out


def rank_and_crowding(objs: list[tuple[float, ...]]) -> tuple[list[int], np.ndarray]:
    fronts = fast_non_dominated_sort(objs)
    ranks = [0] * len(objs)
    distances = np.zeros(len(objs))
    for rank, front in enumerate(fronts):
        d = crowding_distance([objs[i] for i in front])
        for i, idx in enumerate(front):
            ranks[idx] = rank
            distances[idx] = d[i]
    return ranks, distances


def tournament(population, ranks, distances, rng) -> dict:
    i, j = rng.integers(len(population), size=2)
    if ranks[i] < ranks[j] or (ranks[i] == ranks[j] and distances[i] > distances[j]):
        return population[i]["genome"]
    return population[j]["genome"]


def pareto_front(population: list[dict]) -> list[dict]:
    """Front 0 of a population, sorted by error."""
    front = fast_non_dominated_sort(objectives(population))[0]
    return sorted((population[i] for i in front), key=lambda ind: ind["fitness"]["error"])


def run_evolution(
    fitness: Fitness,
    *,
    population_size: int,
    n_generations: int,
    seed: int,
    checkpoint: Path,
    mutation_rate: float = 0.2,
    initial: list[dict] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run (or resume) the search. Returns {"population", "evaluated", "history"}.

    ``initial`` genomes are placed in generation 0 ahead of random ones. ``evaluated`` keeps
    every genome ever trained (with its generation), for the search-cost record.
    """
    rng = np.random.default_rng(seed)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        rng.bit_generator.state = state["rng"]
        log(f"resuming after generation {state['generation']}")
    else:
        genomes = list(initial or [])[:population_size]
        genomes += [sample_genome(rng) for _ in range(population_size - len(genomes))]
        evaluated = [{"genome": g, "fitness": fitness(g, seed), "generation": 0} for g in genomes]
        state = _snapshot(0, evaluated, evaluated, [], rng)
        _summarise(state, log)
        _save(checkpoint, state)

    for generation in range(state["generation"] + 1, n_generations):
        population = state["population"]
        ranks, distances = rank_and_crowding(objectives(population))
        children = []
        for _ in range(population_size):
            a = tournament(population, ranks, distances, rng)
            b = tournament(population, ranks, distances, rng)
            child = mutate(crossover(a, b, rng), rng, rate=mutation_rate)
            children.append(
                {
                    "genome": child,
                    "fitness": fitness(child, seed + generation),
                    "generation": generation,
                }
            )
        pool = population + children
        keep = select_next_generation(objectives(pool), population_size)
        state = _snapshot(
            generation,
            [pool[i] for i in keep],
            state["evaluated"] + children,
            state["history"],
            rng,
        )
        _summarise(state, log)
        _save(checkpoint, state)
    return state


def _snapshot(generation, population, evaluated, history, rng) -> dict:
    errs = [ind["fitness"]["error"] for ind in population]
    finite = [e for e in errs if math.isfinite(e)]
    params = [ind["fitness"]["n_params"] for ind in population]
    history = history + [
        {
            "generation": generation,
            "best_error": min(finite) if finite else math.inf,
            "median_error": float(np.median(finite)) if finite else math.inf,
            "min_params": min(params),
            "max_params": max(params),
            "front_size": len(fast_non_dominated_sort(objectives(population))[0]),
            "evaluations": len(evaluated),
        }
    ]
    return {
        "generation": generation,
        "population": population,
        "evaluated": evaluated,
        "history": history,
        "rng": rng.bit_generator.state,
    }


def _summarise(state, log) -> None:
    h = state["history"][-1]
    log(
        f"generation {h['generation']}: best error {h['best_error']:.4g}, median "
        f"{h['median_error']:.4g}, params {h['min_params']}–{h['max_params']}, "
        f"front {h['front_size']}, evaluations {h['evaluations']}"
    )


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=float))
    tmp.replace(path)
