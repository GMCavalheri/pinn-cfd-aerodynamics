"""RES-01 per-case wrappers and the compute ledger."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pinncfd import benchmark as B
from pinncfd import io
from pinncfd.neuroevolution_search import genome as G

TINY = dict(adam_steps=20, lbfgs_iters=5, eval_every=10, resample_every=10)
CPU = torch.device("cpu")


def test_naca_problem_config_matches_phase2():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
    try:
        from airfoil_flowfield import PROBLEM_CFG
    finally:
        sys.path.pop(0)
    assert B.NACA_PROBLEM_CFG == PROBLEM_CFG


def test_weights_and_config_from_a_genome():
    g = G.PHASE2_GENOME
    assert B.weights_from(g) == {
        "pde": 1.0,
        "body": 10.0,
        "tangency": 1.0,
        "kutta": 10.0,
        "far": 10.0,
        "data": 10.0,
    }
    cfg = B.train_config(g, TINY, seed=3)
    assert cfg.lr == 1e-3 and cfg.lr_final == pytest.approx(5e-5) and cfg.seed == 3


@pytest.mark.parametrize(
    ("case", "genome", "kwargs"),
    [
        ("cylinder", G.PHASE2_GENOME, dict(gamma=0.0, n_interior=256, n_boundary=64)),
        ("joukowski", G.JOUKOWSKI_HAND_GENOME, dict(alpha=0.07, n_interior=128, n_eval=128)),
    ],
)
def test_run_returns_validation_test_times_and_scalars(case, genome, kwargs):
    out = B.run(case, genome, TINY, seed=0, device=CPU, case_kwargs=kwargs)
    assert math.isfinite(out["val"]["error"]) and out["test"] is not None
    assert out["time_train_s"] > 0 and out["n_params"] > 0
    if case == "joukowski":
        assert set(out["scalars"]) == {"gamma", "psi_b_0", "psi_b_1", "psi_b_2"}
        # validation and test are different point sets, so their errors differ
        assert out["val"]["error"] != out["test"]["error"]


def test_a_search_run_never_touches_the_test_set():
    out = B.run(
        "joukowski",
        G.JOUKOWSKI_HAND_GENOME,
        TINY,
        0,
        CPU,
        dict(alpha=0.0, n_interior=128, n_eval=128),
        test=False,
    )
    assert out["test"] is None


def test_a_diverged_run_scores_infinity_and_is_recorded(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("cholesky failed")

    monkeypatch.setattr(B, "train", boom)
    out = B.run("cylinder", G.PHASE2_GENOME, TINY, 0, CPU, dict(n_interior=64, n_boundary=16))
    assert out["val"]["error"] == math.inf and "diverged" in out["stopped"]
    row = B.record("p", "x", out, CPU, tmp_path / "b.csv")
    assert float(row["error"]) == math.inf


def test_record_writes_genome_and_metrics(tmp_path):
    out = B.run(
        "joukowski",
        G.JOUKOWSKI_HAND_GENOME,
        TINY,
        0,
        CPU,
        dict(alpha=0.0, n_interior=128, n_eval=128),
    )
    path = tmp_path / "budget.csv"
    B.record("res01-test", "jk", out, CPU, path)
    (row,) = io.read_budget(path)
    notes = json.loads(row["notes"])
    assert notes["genome"] == G.JOUKOWSKI_HAND_GENOME and notes["case_kwargs"]["alpha"] == 0.0
    metrics = json.loads(row["metrics_json"])
    assert "test_error" in metrics and "scalar_gamma" in metrics


def test_compute_ledger_charges_training_and_evaluation():
    class R:
        time_total_s, time_eval_s = 3.0, 1.0

    ledger = B.ComputeLedger(budget_s=10.0)
    ledger.charge({"result": R(), "val": {"error": 0.1}})
    assert ledger.spent_s == 4.0 and ledger.remaining_s == 6.0 and not ledger.exhausted
    ledger.charge({"result": R(), "val": {"error": 0.2}})
    ledger.charge({"result": None, "val": {"error": np.inf}})  # diverged: nothing measured
    assert ledger.spent_s == 8.0 and ledger.runs == 3
    ledger.charge({"result": R(), "val": {"error": 0.1}})
    assert ledger.exhausted and ledger.remaining_s == 0.0
