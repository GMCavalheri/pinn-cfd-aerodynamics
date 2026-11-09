"""Random search, TPE and ASHA over a gene space, stopped by a compute ledger.

The search space is the same :class:`GeneSpace` neuroevolution uses (frozen genes included),
mapped to Optuna distributions: categorical genes to ``suggest_categorical``, uniform genes to
``suggest_float`` and log-uniform genes to ``suggest_float(log=True)``.

* ``sampler="random"`` — independent uniform draws (Bergstra & Bengio 2012);
* ``sampler="tpe"`` — the tree-structured Parzen estimator (Bergstra et al. 2011), with
  ``n_startup`` random trials first;
* ``pruner="asha"`` — successive halving (Jamieson & Talwalkar 2016; asynchronous form, Li et
  al. 2020) via Optuna's ``SuccessiveHalvingPruner``: each trial reports its validation error at
  every evaluation through the trainer's ``on_eval`` hook and is stopped once it falls in the
  wrong fraction at a rung.

Every trial, pruned or not, is charged to the ledger with its training and evaluation time, and
the study stops after the trial that exhausts the budget, as the NSGA-II loop does.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import optuna

from pinncfd.benchmark import ComputeLedger
from pinncfd.neuroevolution_search.genome import GeneSpace

DIVERGED = 1e9  # Optuna needs a finite value; a diverged trial is ranked last


def suggest(trial: optuna.Trial, space: GeneSpace) -> dict:
    genome = {}
    for name in space.genes:
        if name in space.fixed:
            genome[name] = space.fixed[name]
        elif name in space.choice:
            genome[name] = trial.suggest_categorical(name, space.choice[name])
        elif name in space.uniform:
            genome[name] = trial.suggest_float(name, *space.uniform[name])
        else:
            genome[name] = trial.suggest_float(name, *space.log_uniform[name], log=True)
    return genome


def make_study(sampler: str, pruner: str | None, seed: int, n_startup: int = 10):
    if sampler == "random":
        s = optuna.samplers.RandomSampler(seed=seed)
    elif sampler == "tpe":
        s = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    else:
        raise ValueError("sampler must be 'random' or 'tpe'")
    if pruner is None:
        p = optuna.pruners.NopPruner()
    elif pruner == "asha":
        p = optuna.pruners.SuccessiveHalvingPruner()
    else:
        raise ValueError("pruner must be None or 'asha'")
    return optuna.create_study(direction="minimize", sampler=s, pruner=p)


def run_search(
    evaluate: Callable,
    space: GeneSpace,
    ledger: ComputeLedger,
    *,
    sampler: str = "tpe",
    pruner: str | None = None,
    seed: int = 0,
    n_startup: int = 10,
    max_trials: int | None = None,
    initial: list[dict] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Search until the ledger is exhausted (or ``max_trials``). Returns the best genome, its
    validation error, every trial and the ledger summary. ``initial`` genomes are tried first
    (as NSGA-II seeds generation 0 with the hand-tuned genome)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = make_study(sampler, pruner, seed, n_startup)
    for g in initial or []:
        study.enqueue_trial({k: v for k, v in g.items() if k in space.free})
    trials = []

    def objective(trial: optuna.Trial) -> float:
        genome = suggest(trial, space)

        def on_eval(step: int, row: dict) -> bool:
            if pruner is None:
                return False
            trial.report(float(row["error"]), step)
            return trial.should_prune()

        out = evaluate(genome, seed + trial.number, on_eval)
        ledger.charge(out, label=f"{sampler}{'+' + pruner if pruner else ''} {trial.number}")
        err = out["val"]["error"]
        pruned = "on_eval" in (out.get("stopped") or "")
        trials.append(
            {
                "number": trial.number,
                "genome": genome,
                "error": err,
                "pruned": pruned,
                "n_params": out.get("n_params"),
                "spent_s": ledger.spent_s,
                "stopped": out.get("stopped", ""),
            }
        )
        log(
            f"trial {trial.number:3d}  error {err:.4g}{'  pruned' if pruned else ''}  "
            f"spent {ledger.spent_s:.0f}/{ledger.budget_s:.0f} s"
        )
        if pruned:
            raise optuna.TrialPruned()
        return err if math.isfinite(err) else DIVERGED

    def stop_when_spent(study: optuna.Study, _trial) -> None:
        if ledger.exhausted:
            study.stop()

    study.optimize(objective, n_trials=max_trials, callbacks=[stop_when_spent])
    complete = [t for t in trials if not t["pruned"] and math.isfinite(t["error"])]
    best = min(complete, key=lambda t: t["error"]) if complete else None
    return {
        "best_genome": best["genome"] if best else None,
        "best_error": best["error"] if best else math.inf,
        "trials": trials,
        "ledger": ledger.summary(),
    }
