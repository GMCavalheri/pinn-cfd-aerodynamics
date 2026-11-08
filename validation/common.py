"""Helpers shared by the validation scripts: running a case over seeds and summarising."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pinncfd import io
from pinncfd.nn import n_parameters
from pinncfd.training import convergence, steps_to_target

RESULTS = io.RESULTS
CHECKPOINTS = RESULTS / "checkpoints"
SCRATCH = Path(__file__).resolve().parent / "runs"
EXTRA_TARGETS = (1e-1, 3e-2, 1e-2, 1e-3)


def record(phase, case, seed, model, problem, result, config, device, record_csv=True):
    """Convergence summary + compute-budget row for one finished training run."""
    conv = convergence(result, config)
    metrics = {k: v for k, v in result.final.items() if k not in ("step", "stage", "time_s")}
    for t in EXTRA_TARGETS:
        s = steps_to_target(result.steps, result.errors, t)
        metrics[f"steps_to_{t:.0e}"] = -1 if s is None else s
    row = io.run_row(
        phase,
        case,
        seed,
        result,
        conv,
        metrics,
        "error",
        n_parameters(model),
        problem.n_collocation,
        device,
    )
    if record_csv:
        io.record_run(row)
    return row


def save_histories(path: Path, histories: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {
        key: [{k: (round(v, 8) if isinstance(v, float) else v) for k, v in h.items()} for h in hs]
        for key, hs in histories.items()
    }
    path.write_text(json.dumps(slim, indent=0) + "\n")


def load_histories(path: Path) -> dict:
    return json.loads(path.read_text())


def mean_std(values) -> str:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.3g}"
    return f"{v.mean():.3g} ± {v.std(ddof=1):.2g}"


def fmt_steps(values) -> str:
    """Steps over seeds: 'never' when no seed reached, else min–max range."""
    v = [int(x) for x in values if x not in (None, "", -1, "-1")]
    if not v:
        return "never"
    lo, hi = min(v), max(v)
    note = "" if len(v) == len(values) else f" ({len(v)}/{len(values)} seeds)"
    return (f"{lo}" if lo == hi else f"{lo}–{hi}") + note
