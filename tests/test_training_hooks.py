"""Training-loop hooks for multi-fidelity search and equal-time budgets (RES-01).

A one-parameter problem, loss (p - 1)^2 with error |p - 1|, stands in for a PINN: the hooks
only concern when the loop stops, so the problem can be trivial.
"""

import torch

from pinncfd.training import TrainConfig, train


class Scalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))


class FitOne:
    weights = {"fit": 1.0}

    def losses(self, model):
        return {"fit": (model.p - 1.0).pow(2).sum()}

    def resample(self, rng):
        pass

    def evaluate(self, model):
        return {"error": abs(model.p.item() - 1.0)}


CFG = dict(adam_steps=100, lbfgs_iters=50, eval_every=10, resample_every=0, lr=1e-2)


def test_defaults_run_both_stages_to_the_end():
    res = train(Scalar(), FitOne(), TrainConfig(**CFG))
    assert res.adam_steps == 100 and not res.stopped.startswith(("stopped", "time limit"))
    assert res.history[-1]["stage"] == "lbfgs"


def test_on_eval_stops_adam_and_skips_lbfgs():
    seen = []

    def on_eval(step, row):
        seen.append(step)
        return step >= 30

    res = train(Scalar(), FitOne(), TrainConfig(**CFG), on_eval=on_eval)
    # called at every evaluation after the initial one, and stops at the first True
    assert seen == [10, 20, 30]
    assert res.adam_steps == 30 and res.lbfgs_iters == 0
    assert res.history[-1]["step"] == 30 and "on_eval" in res.stopped
    assert res.stage_of(30) == ("adam", 30)


def test_on_eval_stops_lbfgs():
    def on_eval(step, row):
        return row["stage"] == "lbfgs"

    res = train(Scalar(), FitOne(), TrainConfig(**CFG), on_eval=on_eval)
    assert res.adam_steps == 100 and res.lbfgs_iters >= 1
    assert "L-BFGS iteration" in res.stopped and "on_eval" in res.stopped


def test_time_limit_stops_at_the_first_evaluation_past_it():
    # a zero limit is reached by the first evaluation, whatever the machine's speed
    res = train(Scalar(), FitOne(), TrainConfig(**CFG, time_limit_s=0.0))
    assert res.adam_steps == 10 and res.lbfgs_iters == 0
    assert res.stopped.startswith("time limit")
