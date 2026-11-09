"""RES-01 tuning baselines: Optuna random search, TPE, ASHA, and the equal-budget NSGA-II.

The objective is closed-form, so these tests need no training: a fake ``evaluate`` returns
what ``benchmark.run`` returns, with an error that is smallest at lr = 1e-3 and w_body = 10,
and a fixed cost of 1 s per run.
"""

import math

import numpy as np
import pytest

from pinncfd.benchmark import ComputeLedger
from pinncfd.hpo.nsga_budget import run_nsga
from pinncfd.hpo.optuna_search import make_study, run_search, suggest
from pinncfd.neuroevolution_search import genome as G

SPACE = G.JOUKOWSKI_SPACE


class Res:
    time_total_s, time_eval_s = 1.0, 0.0


def target(g):
    return (np.log10(g["lr"]) + 3) ** 2 + (np.log10(g["w_body"]) - 1) ** 2


def fake_evaluate(g, seed, on_eval):
    err = float(target(g))
    stopped = ""
    if on_eval is not None:
        # a learning curve that ends at the target value: error(step) = err + 1/step
        for step in range(10, 110, 10):
            if on_eval(step, {"error": err + 1.0 / step}):
                stopped = f"stopped by on_eval at Adam step {step}"
                break
    return {"val": {"error": err}, "result": Res(), "stopped": stopped, "n_params": 100}


def test_suggestions_stay_in_the_domain_and_respect_frozen_genes():
    space = G.ablation_space(SPACE, G.JOUKOWSKI_HAND_GENOME, ["loss_and_sampling"])
    study = make_study("random", None, seed=0)
    for _ in range(50):
        g = suggest(study.ask(), space)
        assert space.in_domain(g)
        assert g["width"] == G.JOUKOWSKI_HAND_GENOME["width"]  # architecture frozen


def test_tpe_improves_on_its_random_start():
    res = run_search(
        fake_evaluate,
        SPACE,
        ComputeLedger(1e9),
        sampler="tpe",
        seed=0,
        max_trials=60,
        log=lambda s: None,
    )
    errs = [t["error"] for t in res["trials"]]
    # after 10 random start-up trials TPE models the objective: its best is far below the
    # best of the random phase (the minimum is 0 at lr = 1e-3, w_body = 10)
    assert min(errs[10:]) < 0.25 * min(errs[:10])
    assert res["best_error"] == min(errs)


def test_asha_prunes_trials_and_charges_them():
    ledger = ComputeLedger(1e9)
    res = run_search(
        fake_evaluate,
        SPACE,
        ledger,
        sampler="random",
        pruner="asha",
        seed=1,
        max_trials=40,
        log=lambda s: None,
    )
    pruned = [t for t in res["trials"] if t["pruned"]]
    assert 0 < len(pruned) < 40
    assert ledger.runs == 40  # pruned trials cost compute too
    assert all(not math.isinf(t["error"]) for t in res["trials"])


def test_the_study_stops_when_the_budget_is_spent():
    ledger = ComputeLedger(budget_s=7.0)
    res = run_search(fake_evaluate, SPACE, ledger, sampler="random", seed=0, log=lambda s: None)
    # 1 s per trial: the seventh trial exhausts the budget and the study stops after it
    assert len(res["trials"]) == 7 and ledger.exhausted


def test_initial_genomes_are_tried_first():
    hand = G.JOUKOWSKI_HAND_GENOME
    res = run_search(
        fake_evaluate,
        SPACE,
        ComputeLedger(1e9),
        sampler="tpe",
        seed=0,
        max_trials=3,
        initial=[hand],
        log=lambda s: None,
    )
    assert res["trials"][0]["genome"] == hand


def test_nsga_runs_whole_generations_until_the_budget_is_spent(tmp_path):
    ledger = ComputeLedger(budget_s=20.0)
    res = run_nsga(
        fake_evaluate,
        SPACE,
        ledger,
        population_size=6,
        seed=0,
        checkpoint=tmp_path / "ck.json",
        initial=[G.JOUKOWSKI_HAND_GENOME],
        log=lambda s: None,
    )
    # 6 genomes per generation at 1 s each: generations 0-3 spend 24 s, the first total >= 20
    assert res["generations"] == 4 and ledger.spent_s == 24.0
    assert len(res["evaluated"]) == 24
    assert all(SPACE.in_domain(e["genome"]) for e in res["evaluated"])
    assert res["best_error"] == pytest.approx(min(e["fitness"]["error"] for e in res["evaluated"]))
