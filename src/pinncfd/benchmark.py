"""One entry point per RES-01 case: build, train and score a PINN from a genome, and account
for compute.

RES-01 compares several ways of choosing a PINN's configuration (hand tuning, neuroevolution,
random search, TPE, ASHA) on the same cases under the same compute budget. Every method calls
:func:`run` with a genome (a dict of genes, see ``neuroevolution_search.genome``), a training
budget and a seed, and gets back the validation and test metrics, the time spent and the
learned scalars. :class:`ComputeLedger` adds up what each method spends, search included.

Cases (``CASES``):

* ``"cylinder"`` — Phase 1's hard-constraint stream-function PINN (``case_kwargs``: ``gamma``).
  The calibration case: its single evaluation grid serves as both validation and test.
* ``"joukowski"`` — RES-01 case 2 (``case_kwargs``: ``alpha`` [rad]); genome from
  ``JOUKOWSKI_SPACE``.
* ``"naca_family"`` — the Phase 2-3 airfoil-family network (case 3); genome from
  ``PHASE3_SPACE``. Validation: the 8 validation cases; test: the 24 test cases.
* ``"airdata"`` — the inverse problem (``pinncfd.inverse_airdata``), registered by that
  module.

Diverged runs (non-finite error, or a runtime error such as a failed Cholesky in L-BFGS) return
an infinite validation error, as in the Phase 3 search.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from pinncfd import io
from pinncfd.nn import n_parameters
from pinncfd.seed import seed_everything
from pinncfd.training import TrainConfig, convergence, train

RES01_BUDGET = io.ROOT / "experiments" / "res01" / "results" / "compute_budget.csv"

# validation/airfoil_flowfield.py PROBLEM_CFG (the Phase 2-3 problem); tests check they agree
NACA_PROBLEM_CFG = dict(
    n_cases=32, n_interior=384, n_body=128, n_far=64, n_data_cases=16, n_data=128
)


def weights_from(genome: dict) -> dict:
    """Loss weights from the w_* genes, with the PDE weight 1 (only ratios matter)."""
    return {"pde": 1.0} | {k[2:]: v for k, v in genome.items() if k.startswith("w_")}


def train_config(genome: dict, budget: dict, seed: int) -> TrainConfig:
    """The budget plus the genome's learning rate, decayed 20x over Adam (as in Phase 3)."""
    lr = genome.get("lr", 1e-3)
    return TrainConfig(**dict(budget, lr=lr, lr_final=lr / 20), seed=seed)


def mlp_from(genome: dict, seed: int) -> dict:
    keys = ("width", "depth", "activation", "fourier_features", "fourier_scale")
    return {k: genome[k] for k in keys if k in genome} | {"seed": seed}


# ---- builders ----------------------------------------------------------------------------------


def _build_cylinder(genome, seed, device, gamma: float = 0.0, **problem_kw):
    from pinncfd.pinn_laplace_2d import CylinderPINN, CylinderProblem

    model = CylinderPINN("psi", hard=True, gamma=gamma, mlp=mlp_from(genome, seed)).to(device)
    kw = dict(n_interior=genome.get("n_interior", 8192)) | problem_kw
    return model, CylinderProblem(model, device=device, seed=seed, **kw)


def _build_joukowski(genome, seed, device, alpha: float = 0.0, **problem_kw):
    from pinncfd.pinn_joukowski import JoukowskiPINN, JoukowskiProblem

    model = JoukowskiPINN(alpha, genome.get("features", "plain"), mlp_from(genome, seed))
    model = model.to(device)
    kw = dict(n_interior=genome["n_interior"], weights=weights_from(genome)) | problem_kw
    return model, JoukowskiProblem(model, device=device, seed=seed, **kw)


@lru_cache(maxsize=2)
def naca_references():
    """(validation, test) panel references of the airfoil family, built once per process."""
    from pinncfd.pinn_airfoil_flowfield import build_reference, family

    return (
        build_reference(family.val_cases(), seed=1),
        build_reference(family.test_cases(), seed=2),
    )


def _build_naca(genome, seed, device, **problem_kw):
    from pinncfd.neuroevolution_search.fitness import build_model, build_problem

    val_ref, _ = naca_references()
    model = build_model(genome, seed).to(device)
    cfg = NACA_PROBLEM_CFG | problem_kw
    return model, build_problem(genome, model, val_ref, device, seed, cfg)


BUILDERS: dict[str, Callable] = {
    "cylinder": _build_cylinder,
    "joukowski": _build_joukowski,
    "naca_family": _build_naca,
}
CASES = BUILDERS  # "airdata" is added by pinncfd.inverse_airdata


def build(case: str, genome: dict, seed: int, device, case_kwargs: dict | None = None):
    """(model, problem) for ``case`` from ``genome``."""
    if case == "airdata" and case not in BUILDERS:
        import pinncfd.inverse_airdata  # noqa: F401  (registers the builder)
    return BUILDERS[case](genome, seed, device, **(case_kwargs or {}))


# ---- evaluation --------------------------------------------------------------------------------


def evaluate_test(case: str, model, problem) -> dict:
    if case == "naca_family":
        return problem.evaluate(model, naca_references()[1])
    if case == "cylinder":
        return problem.evaluate(model)
    return problem.evaluate(model, split="test")


def learned_scalars(model) -> dict:
    """Scalars a model learns besides its network (Gamma, psi_b, alpha, U), as floats."""
    out = {}
    for name in ("gamma", "psi_b", "alpha_hat", "log_u"):
        p = getattr(model, name, None)
        if p is not None and hasattr(p, "detach"):
            out[name] = float(p.detach())
    if "log_u" in out:
        out["u_hat"] = math.exp(out.pop("log_u"))
    return out


def run(
    case: str,
    genome: dict,
    budget: dict,
    seed: int,
    device,
    case_kwargs: dict | None = None,
    on_eval: Callable[[int, dict], bool] | None = None,
    test: bool = True,
) -> dict:
    """Train ``case`` from ``genome`` with ``budget`` (TrainConfig fields) and score it.

    Returns a dict with ``val`` (the last validation metrics), ``test`` (None unless ``test``),
    times, sizes, learned scalars and the stop reason, plus the live ``model``, ``problem``,
    ``result`` and ``config`` for checkpoints and compute records. A search passes
    ``test=False``: it must never see the test set.
    """
    seed_everything(seed)
    model, problem = build(case, genome, seed, device, case_kwargs)
    cfg = train_config(genome, budget, seed)
    out = {
        "case": case,
        "case_kwargs": dict(case_kwargs or {}),
        "genome": dict(genome),
        "seed": seed,
        "n_params": n_parameters(model),
        "n_collocation": problem.n_collocation,
        "model": model,
        "problem": problem,
        "config": cfg,
    }
    try:
        result = train(model, problem, cfg, device, on_eval=on_eval)
        if not math.isfinite(result.final["error"]):
            raise FloatingPointError("non-finite validation error")
    except (FloatingPointError, RuntimeError) as exc:
        return out | {
            "val": {"error": math.inf},
            "test": None,
            "result": None,
            "time_train_s": math.nan,
            "time_eval_s": math.nan,
            "scalars": {},
            "stopped": f"diverged: {str(exc)[:200]}",
        }
    final = {k: v for k, v in result.final.items() if k not in ("step", "stage", "time_s")}
    return out | {
        "val": final,
        "test": evaluate_test(case, model, problem) if test else None,
        "result": result,
        "time_train_s": result.time_total_s,
        "time_eval_s": result.time_eval_s,
        "scalars": learned_scalars(model),
        "stopped": result.stopped,
    }


def record(phase: str, name: str, out: dict, device, path=RES01_BUDGET) -> dict:
    """Append a finished run to a compute-budget CSV (default: RES-01's own record)."""
    if out["result"] is None:
        row = {
            "phase": phase,
            "case": name,
            "seed": out["seed"],
            "error_name": "error",
            "error": math.inf,
            "n_params": out["n_params"],
            "notes": _notes(out),
        }
        return io.record_run(row, path)
    res, cfg = out["result"], out["config"]
    metrics = dict(out["val"])
    if out["test"] is not None:
        metrics |= {f"test_{k}": v for k, v in out["test"].items()}
    metrics |= {f"scalar_{k}": v for k, v in out["scalars"].items()}
    row = io.run_row(
        phase,
        name,
        out["seed"],
        res,
        convergence(res, cfg),
        metrics,
        "error",
        out["n_params"],
        out["n_collocation"],
        device,
    )
    row["notes"] = _notes(out)
    return io.record_run(row, path)


def _notes(out: dict) -> str:
    """The stop reason, genome and case parameters, as JSON (the CSV's free-text column)."""
    return json.dumps(
        {"stopped": out["stopped"], "genome": out["genome"], "case_kwargs": out["case_kwargs"]}
    )


# ---- compute accounting ------------------------------------------------------------------------


@dataclass
class ComputeLedger:
    """What one method has spent, in seconds of device time: training plus evaluation.

    Evaluation is charged too: a search that scores its candidates more often pays for it.
    """

    budget_s: float
    train_s: float = 0.0
    eval_s: float = 0.0
    runs: int = 0
    log: list = field(default_factory=list)

    @property
    def spent_s(self) -> float:
        return self.train_s + self.eval_s

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.budget_s - self.spent_s)

    @property
    def exhausted(self) -> bool:
        return self.spent_s >= self.budget_s

    def charge(self, out: dict, label: str = "") -> None:
        """Charge a :func:`run` result (a diverged run is charged its wall time if known)."""
        res = out.get("result")
        t, e = (res.time_total_s, res.time_eval_s) if res is not None else (0.0, 0.0)
        self.train_s += t
        self.eval_s += e
        self.runs += 1
        self.log.append({"label": label, "train_s": t, "eval_s": e, "error": out["val"]["error"]})

    def summary(self) -> dict:
        return {
            "budget_s": self.budget_s,
            "spent_s": self.spent_s,
            "train_s": self.train_s,
            "eval_s": self.eval_s,
            "runs": self.runs,
        }


def errors(outs) -> np.ndarray:
    return np.array([o["val"]["error"] for o in outs], dtype=float)
