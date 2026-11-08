"""NSGA-II's two core mechanisms: non-dominated sorting and crowding distance.

Ported unchanged in substance from the RUL project (GMCavalheri/Remaining-Useful-Life-RUL-
Prediction-with-PINN-and-Neuroevolution, ``src/rul/evolution/nsga.py`` at commit addc209), the
same evolutionary machinery the plan asks this project to reuse.

The search minimises two competing objectives per genome: validation error (accuracy) and
parameter count (inference cost). Genome A dominates genome B if A is no worse on every
objective and strictly better on at least one; there is then no single best genome but a
*Pareto front* of non-dominated ones.

**Non-dominated sorting** (:func:`fast_non_dominated_sort`, Deb et al. 2002) partitions a
population into fronts: front 0 is every genome no one dominates, front 1 every genome
dominated only by front 0, and so on. The next generation is filled front by front.

**Crowding distance** (:func:`crowding_distance`) breaks ties inside the front that has to be
truncated, preferring genomes from sparse regions so the population stays spread along the
whole trade-off curve.
"""

from __future__ import annotations

import numpy as np


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """True if objective vector ``a`` dominates ``b`` (both minimized)."""
    pairs = list(zip(a, b, strict=True))
    return all(x <= y for x, y in pairs) and any(x < y for x, y in pairs)


def fast_non_dominated_sort(objectives: list[tuple[float, ...]]) -> list[list[int]]:
    """Partition indices into Pareto fronts, front 0 = non-dominated."""
    n = len(objectives)
    domination_counts = [0] * n  # how many individuals dominate i
    dominated_by = [[] for _ in range(n)]  # indices i dominates

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                domination_counts[i] += 1

    fronts: list[list[int]] = []
    current_front = [i for i in range(n) if domination_counts[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_by[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def crowding_distance(front_objectives: list[tuple[float, ...]]) -> np.ndarray:
    """Crowding distance for each individual in a single front.

    For each objective, sort the front by that objective and accumulate the
    (normalized) gap to each individual's neighbors; boundary individuals
    (best/worst on that objective) get infinite distance so they are always
    kept. Individuals in sparser regions of the front get a larger total
    distance.
    """
    n = len(front_objectives)
    if n == 0:
        return np.array([])
    if n <= 2:
        return np.full(n, np.inf)

    n_objectives = len(front_objectives[0])
    distances = np.zeros(n)
    obj_array = np.array(front_objectives)

    for k in range(n_objectives):
        order = np.argsort(obj_array[:, k])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        obj_range = obj_array[order[-1], k] - obj_array[order[0], k]
        # obj_range is 0 when every individual is identical on this
        # objective (no information), and can be +inf when the front mixes
        # a finite value with an unfit individual's objective (e.g.
        # a diverged training run, whose error is recorded as inf). Either way there's no
        # meaningful *relative* gap to measure, so this objective
        # contributes nothing rather than a NaN from inf-inf or x/inf.
        if obj_range == 0 or not np.isfinite(obj_range):
            continue

        for idx in range(1, n - 1):
            prev_val = obj_array[order[idx - 1], k]
            next_val = obj_array[order[idx + 1], k]
            gap = next_val - prev_val
            if np.isfinite(gap):
                distances[order[idx]] += gap / obj_range

    return distances


def select_next_generation(objectives: list[tuple[float, ...]], population_size: int) -> list[int]:
    """Indices of the ``population_size`` individuals to keep, via NSGA-II selection.

    Fronts are added whole while they fit; the last (partially included)
    front is truncated by descending crowding distance, so the kept portion
    favors individuals from the sparsest, most spread-out regions of that
    front.
    """
    fronts = fast_non_dominated_sort(objectives)
    selected: list[int] = []

    for front in fronts:
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
        else:
            remaining = population_size - len(selected)
            front_objectives = [objectives[i] for i in front]
            distances = crowding_distance(front_objectives)
            order = np.argsort(-distances)  # descending: most spread-out first
            selected.extend(front[i] for i in order[:remaining])
            break

    return selected
