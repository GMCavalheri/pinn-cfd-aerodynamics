"""Training loop shared by every PINN in the project: Adam, then L-BFGS, with tracking.

The two-stage optimiser is the standard recipe for PINNs (Raissi et al. 2019; Wang et al.
2021): Adam finds the basin from a random start, where L-BFGS would stall on the rough early
landscape; L-BFGS then converges far faster inside the basin, where the loss is smooth and
nearly quadratic.

Besides the loss, the loop evaluates the problem's own error metric every ``eval_every`` steps,
which gives the two convergence numbers the compute-budget record needs:

* **steps to target** — the first evaluated step at which the error is at or below a target;
* **plateau step** — the first evaluated step after which the best error so far never again
  improves by more than ``plateau_tol`` (relative) within ``plateau_window`` steps.

Steps are counted globally: Adam steps 1..A, then L-BFGS iterations A+1..A+L. Each record also
says which stage the step falls in. Evaluation time is excluded from the training time.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch


class Problem(Protocol):
    """What the trainer needs from a PINN problem."""

    weights: dict[str, float]

    def losses(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Unweighted mean-square loss terms, keyed like ``weights``."""

    def resample(self, rng: np.random.Generator) -> None:
        """Redraw the collocation points (called during the Adam stage only)."""

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        """Error metrics against the reference; must contain the key ``"error"``."""


@dataclass
class TrainConfig:
    adam_steps: int = 5000
    lr: float = 1e-3
    lr_final: float = 1e-4  # exponential decay from lr to lr_final over the Adam stage
    lbfgs_iters: int = 2000
    lbfgs_history: int = 50
    eval_every: int = 100
    resample_every: int = 500
    target_error: float = 1e-2
    plateau_window: int = 500
    plateau_tol: float = 0.01
    seed: int = 0
    # stop once the training time reaches this many seconds (checked at every evaluation, so
    # it can overrun by up to one ``eval_every`` chunk); None = no limit. For equal-time
    # budgets across methods.
    time_limit_s: float | None = None


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    time_adam_s: float = 0.0
    time_lbfgs_s: float = 0.0
    time_eval_s: float = 0.0
    adam_steps: int = 0
    lbfgs_iters: int = 0
    stopped: str = ""  # why training stopped early, if it did

    @property
    def time_total_s(self) -> float:
        return self.time_adam_s + self.time_lbfgs_s

    @property
    def steps(self) -> np.ndarray:
        return np.array([h["step"] for h in self.history])

    @property
    def errors(self) -> np.ndarray:
        return np.array([h["error"] for h in self.history])

    @property
    def final(self) -> dict:
        return self.history[-1]

    def stage_of(self, step: int | None) -> tuple[str, int] | tuple[None, None]:
        if step is None:
            return None, None
        if step <= self.adam_steps:
            return "adam", step
        return "lbfgs", step - self.adam_steps


def steps_to_target(steps, errors, target: float) -> int | None:
    """First evaluated step whose error is <= ``target``; None if never reached."""
    hit = np.flatnonzero(np.asarray(errors) <= target)
    return int(np.asarray(steps)[hit[0]]) if len(hit) else None


def plateau_step(steps, errors, window: int, tol: float) -> int:
    """First step after which the running-best error never improves by > ``tol`` per window.

    For each evaluated step s, compare the best error up to s with the best error up to
    s + window (the final best when the window runs past the end). The plateau is the first s
    from which every later comparison shows a relative improvement of at most ``tol``.
    """
    steps = np.asarray(steps, dtype=float)
    best = np.minimum.accumulate(np.asarray(errors, dtype=float))
    later_idx = np.searchsorted(steps, steps + window, side="right") - 1
    improving = 1.0 - best[later_idx] / best > tol
    if not improving.any():
        return int(steps[0])
    last = np.flatnonzero(improving)[-1]
    return int(steps[min(last + 1, len(steps) - 1)])


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _over_time(res: TrainResult, config: TrainConfig) -> bool:
    return config.time_limit_s is not None and res.time_total_s >= config.time_limit_s


def _halt_reason(by_callback: bool, config: TrainConfig, where: str) -> str:
    if by_callback:
        return f"stopped by on_eval at {where}"
    return f"time limit {config.time_limit_s:g} s reached at {where}"


def train(
    model: torch.nn.Module,
    problem: Problem,
    config: TrainConfig,
    device=None,
    on_eval: Callable[[int, dict], bool] | None = None,
) -> TrainResult:
    """Train ``model`` on ``problem``; returns the evaluated history and the timings.

    ``on_eval(step, row)`` is called after every evaluation after the first (``row`` is the
    history entry just logged); returning True stops training there, e.g. a pruned trial in a
    multi-fidelity search. With ``config.time_limit_s`` training also stops at the first
    evaluation at which the training time has reached the limit. Either way ``stopped`` says
    why, and an early stop in the Adam stage skips L-BFGS.
    """
    device = device or next(model.parameters()).device
    rng = np.random.default_rng(config.seed)
    res = TrainResult()
    params = [p for p in model.parameters() if p.requires_grad]

    def total_loss():
        terms = problem.losses(model)
        loss = sum(problem.weights[k] * v for k, v in terms.items())
        return loss, terms

    def log(step: int, loss: float, terms: dict) -> bool:
        _sync(device)
        t0 = time.perf_counter()
        metrics = problem.evaluate(model)
        _sync(device)
        res.time_eval_s += time.perf_counter() - t0
        stage, _ = res.stage_of(step) if step else ("init", 0)
        row = {"step": step, "stage": stage, "loss": loss}
        row.update({f"loss_{k}": v.item() for k, v in terms.items()})
        row.update(metrics)
        row["time_s"] = res.time_adam_s + res.time_lbfgs_s
        res.history.append(row)
        return bool(step and on_eval is not None and on_eval(step, row))

    # ---- Adam -------------------------------------------------------------------------------
    opt = torch.optim.Adam(params, lr=config.lr)
    gamma = (config.lr_final / config.lr) ** (1.0 / max(config.adam_steps, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    res.adam_steps = config.adam_steps
    loss, terms = total_loss()
    log(0, loss.item(), terms)
    _sync(device)
    t0 = time.perf_counter()
    halted = False
    for step in range(1, config.adam_steps + 1):
        if config.resample_every and step % config.resample_every == 0:
            problem.resample(rng)
        opt.zero_grad(set_to_none=True)
        loss, terms = total_loss()
        loss.backward()
        opt.step()
        sched.step()
        if step % config.eval_every == 0 or step == config.adam_steps:
            _sync(device)
            res.time_adam_s += time.perf_counter() - t0
            by_callback = log(step, loss.item(), terms)
            if by_callback or _over_time(res, config):
                res.adam_steps = step
                res.stopped = _halt_reason(by_callback, config, f"Adam step {step}")
                halted = True
                break
            t0 = time.perf_counter()
    if not halted:
        _sync(device)
        res.time_adam_s += time.perf_counter() - t0

    # ---- L-BFGS -----------------------------------------------------------------------------
    if config.lbfgs_iters > 0 and not halted:
        opt = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=config.eval_every,
            history_size=config.lbfgs_history,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-15,
        )
        last = {"loss": None, "terms": None}

        def closure():
            opt.zero_grad(set_to_none=True)
            loss, terms = total_loss()
            loss.backward()
            last["loss"], last["terms"] = loss, terms
            return loss

        done = 0
        good_state = copy.deepcopy(model.state_dict())
        while done < config.lbfgs_iters:
            chunk = min(config.eval_every, config.lbfgs_iters - done)
            # max_eval would otherwise default to 1.25 max_iter and end a chunk early, which
            # the check below would mistake for convergence
            opt.param_groups[0].update(max_iter=chunk, max_eval=10 * chunk)
            _sync(device)
            t0 = time.perf_counter()
            opt.step(closure)
            _sync(device)
            res.time_lbfgs_s += time.perf_counter() - t0
            n_iter = opt.state[params[0]].get("n_iter", 0)
            advanced = n_iter - done
            done = n_iter
            loss = last["loss"].item()
            if not math.isfinite(loss):
                model.load_state_dict(good_state)
                res.stopped = f"non-finite loss after {done} L-BFGS iterations; rolled back"
                break
            res.lbfgs_iters = done
            by_callback = log(res.adam_steps + done, loss, last["terms"])
            good_state = copy.deepcopy(model.state_dict())
            if by_callback or _over_time(res, config):
                res.stopped = _halt_reason(by_callback, config, f"L-BFGS iteration {done}")
                break
            if advanced < opt.param_groups[0]["max_iter"]:
                res.stopped = f"L-BFGS converged (tolerance) after {done} iterations"
                break
    return res


def convergence(res: TrainResult, config: TrainConfig) -> dict:
    """Steps-to-target and plateau, as global step plus (stage, step within stage)."""
    target = steps_to_target(res.steps, res.errors, config.target_error)
    plateau = plateau_step(res.steps, res.errors, config.plateau_window, config.plateau_tol)
    t_stage, t_step = res.stage_of(target)
    p_stage, p_step = res.stage_of(plateau)
    time_at = {h["step"]: h["time_s"] for h in res.history}
    return {
        "target_error": config.target_error,
        "steps_to_target": target,
        "steps_to_target_stage": t_stage,
        "steps_to_target_in_stage": t_step,
        "time_to_target_s": time_at.get(target),
        "plateau_window": config.plateau_window,
        "plateau_tol": config.plateau_tol,
        "plateau_step": plateau,
        "plateau_stage": p_stage,
        "plateau_in_stage": p_step,
        "time_to_plateau_s": time_at.get(plateau),
    }
