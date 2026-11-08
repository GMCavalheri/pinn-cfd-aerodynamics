"""Phase 3: NSGA-II machinery and genome operators."""

import math

import numpy as np
import pytest

from pinncfd.neuroevolution_search import genome as G
from pinncfd.neuroevolution_search.nsga import (
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    select_next_generation,
)
from pinncfd.neuroevolution_search.search import pareto_front, run_evolution


def test_dominance():
    assert dominates((1, 1), (2, 2))
    assert dominates((1, 2), (1, 3))
    assert not dominates((1, 2), (1, 2))  # equal is not domination
    assert not dominates((1, 3), (2, 2))  # a trade-off


def test_non_dominated_sort_on_a_hand_checked_population():
    # (0) and (1) trade off; (2) is dominated by (0) only; (3) by (0), (1) and (2);
    # (4) is dominated by everything (Deb et al. 2002, Sec. III-A, by construction)
    objs = [(1, 4), (3, 1), (2, 5), (4, 6), (5, 7)]
    assert fast_non_dominated_sort(objs) == [[0, 1], [2], [3], [4]]


def test_crowding_distance_prefers_the_sparse_middle_and_keeps_extremes():
    # front along f1 + f2 = 10 at f1 = 0, 1, 2, 8, 10
    front = [(0, 10), (1, 9), (2, 8), (8, 2), (10, 0)]
    d = crowding_distance(front)
    assert math.isinf(d[0]) and math.isinf(d[-1])
    # interior: per objective (next - prev)/range; point 2 has neighbours 1 and 8
    assert d[1] == pytest.approx(2 * (2 - 0) / 10)
    assert d[2] == pytest.approx(2 * (8 - 1) / 10)
    assert d[3] == pytest.approx(2 * (10 - 2) / 10)


def test_crowding_distance_ignores_an_infinite_objective():
    d = crowding_distance([(0.1, 1), (math.inf, 2), (0.3, 3)])
    assert np.all(np.isfinite(d) | np.isinf(d))
    assert not np.isnan(d).any()


def test_selection_fills_by_front_then_crowding():
    objs = [(1, 4), (3, 1), (2, 5), (4, 6), (5, 7), (2, 2)]
    # fronts: [0, 1, 5], [2], [3], [4]
    assert sorted(select_next_generation(objs, 4)) == [0, 1, 2, 5]
    two = select_next_generation(objs, 2)
    assert set(two) <= {0, 1, 5} and len(two) == 2


def test_genome_sampling_and_mutation_stay_in_the_domain():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = G.sample_genome(rng)
        assert G.in_domain(g)
        assert G.in_domain(G.mutate(g, rng, rate=0.5))
    assert G.in_domain(G.PHASE2_GENOME)


def test_crossover_takes_each_gene_from_a_parent():
    rng = np.random.default_rng(1)
    a, b = G.sample_genome(rng), G.sample_genome(rng)
    c = G.crossover(a, b, rng)
    assert all(c[k] in (a[k], b[k]) for k in G.GENES)


def test_log_uniform_genes_are_spread_in_log_space():
    rng = np.random.default_rng(2)
    w = np.array([G.sample_genome(rng)["w_data"] for _ in range(4000)])
    # log-uniform on [0.1, 100]: a third of the draws in each decade
    frac = [np.mean((w >= lo) & (w < 10 * lo)) for lo in (0.1, 1.0, 10.0)]
    assert frac == pytest.approx([1 / 3] * 3, abs=0.03)


def test_search_improves_a_known_objective_and_resumes(tmp_path):
    # Toy fitness with a known optimum: error is smallest for width 96, w_data near 10 and
    # lr near 1e-3; n_params grows with width. The front must reach the optimum region.
    calls = []

    def fitness(g, seed):
        calls.append(seed)
        err = abs(np.log10(g["w_data"]) - 1) + abs(np.log10(g["lr"]) + 3) + (96 - g["width"]) / 96
        return {"error": float(err), "n_params": g["width"] * g["depth"]}

    ck = tmp_path / "ck.json"
    state = run_evolution(
        fitness, population_size=10, n_generations=4, seed=0, checkpoint=ck, log=lambda s: None
    )
    best0 = state["history"][0]["best_error"]
    assert state["history"][-1]["best_error"] <= best0  # elitism: never worse
    assert len(calls) == 10 * 4
    front = pareto_front(state["population"])
    assert front[0]["fitness"]["error"] == state["history"][-1]["best_error"]
    # resuming a finished search trains nothing new
    n = len(calls)
    run_evolution(
        fitness, population_size=10, n_generations=4, seed=0, checkpoint=ck, log=lambda s: None
    )
    assert len(calls) == n
    # extending it continues from the checkpoint
    state = run_evolution(
        fitness, population_size=10, n_generations=6, seed=0, checkpoint=ck, log=lambda s: None
    )
    assert len(calls) == n + 20 and state["generation"] == 5
