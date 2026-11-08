"""Fitness of a genome: train the airfoil-family PINN it describes with a short proxy budget.

Every genome gets the same budget (Adam steps; no L-BFGS, whose run length depends on the loss
landscape and would make costs incomparable), the same validation cases and the same panel
reference. Its fitness is (validation velocity error, parameter count). A run that diverges
gets an infinite error, which NSGA-II ranks last.

Short-budget fitness is a *proxy*: it favours genomes that learn fast, not necessarily those
that end best after a long run. Phase 3 therefore retrains the front's champions with the full
Phase 2 budget and compares them on the test set, which the search never sees.
"""

from __future__ import annotations

import math
import time

from pinncfd.nn import n_parameters
from pinncfd.pinn_airfoil_flowfield import AirfoilPINN, AirfoilProblem
from pinncfd.seed import seed_everything
from pinncfd.training import TrainConfig, train

HEAD = dict(width=32, depth=2, activation="tanh")


def build_model(genome: dict, seed: int = 0) -> AirfoilPINN:
    mlp = dict(
        width=genome["width"],
        depth=genome["depth"],
        activation=genome["activation"],
        fourier_features=genome["fourier_features"],
        fourier_scale=genome["fourier_scale"],
        seed=seed,
    )
    return AirfoilPINN(mlp=mlp, head=dict(HEAD, seed=seed))


def weights(genome: dict) -> dict:
    return {
        "pde": 1.0,
        "body": genome["w_body"],
        "tangency": genome["w_tangency"],
        "kutta": genome["w_kutta"],
        "far": genome["w_far"],
        "data": genome["w_data"],
    }


def build_problem(genome, model, reference, device, seed, problem_cfg: dict) -> AirfoilProblem:
    cfg = dict(problem_cfg, n_interior=genome["n_interior"])
    return AirfoilProblem(
        model, evaluation=reference, weights=weights(genome), device=device, seed=seed, **cfg
    )


def train_config(genome: dict, base: dict, seed: int) -> TrainConfig:
    return TrainConfig(**dict(base, lr=genome["lr"], lr_final=genome["lr"] / 20), seed=seed)


def evaluate_genome(genome, reference, device, seed, budget: dict, problem_cfg: dict):
    """Train with the proxy budget; returns fitness plus the run's cost and history."""
    seed_everything(seed)
    model = build_model(genome, seed).to(device)
    problem = build_problem(genome, model, reference, device, seed, problem_cfg)
    cfg = train_config(genome, budget, seed)
    t0 = time.perf_counter()
    try:
        result = train(model, problem, cfg, device)
        err = result.final["error"]
        if not math.isfinite(err):
            raise FloatingPointError("non-finite validation error")
    except (FloatingPointError, RuntimeError) as exc:
        return (
            {
                "error": math.inf,
                "n_params": n_parameters(model),
                "time_s": time.perf_counter() - t0,
                "failed": str(exc)[:200],
            },
            None,
            None,
            None,
            cfg,
        )
    fit = {
        "error": float(err),
        "n_params": n_parameters(model),
        "time_s": result.time_total_s,
        "cp_rms": result.final["cp_rms"],
        "cl_err_circulation": result.final["cl_err_circulation"],
    }
    return fit, model, problem, result, cfg
