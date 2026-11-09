"""Checkpoints and the compute-budget record.

Every training run appends one row to ``validation/results/compute_budget.csv``: its error,
wall-clock training time per optimiser stage, steps to the target error and to the plateau,
and enough metadata (device, seed, commit, library versions) to reproduce it. A rerun of the
same (phase, case, seed) replaces its row instead of adding a duplicate.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation" / "results"
BUDGET_CSV = RESULTS / "compute_budget.csv"

COLUMNS = [
    "phase",
    "case",
    "seed",
    "error_name",
    "error",
    "target_error",
    "steps_to_target",
    "steps_to_target_stage",
    "steps_to_target_in_stage",
    "time_to_target_s",
    "plateau_window",
    "plateau_tol",
    "plateau_step",
    "plateau_stage",
    "plateau_in_stage",
    "time_to_plateau_s",
    "adam_steps",
    "lbfgs_iters",
    "time_adam_s",
    "time_lbfgs_s",
    "time_total_s",
    "n_params",
    "n_collocation",
    "device",
    "torch",
    "cuda",
    "git_commit",
    "date",
    "metrics_json",
    "notes",
]


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        # results written by the runs themselves (this CSV included) do not make the code dirty
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                ".",
                ":(exclude)validation/results",
                ":(exclude)experiments/res01/results",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        return "unknown"
    return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"


def device_name(device) -> str:
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"cpu ({platform.processor() or platform.machine()})"


def environment(device) -> dict:
    return {
        "device": device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "git_commit": git_commit(),
        "date": dt.date.today().isoformat(),
    }


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def record_run(row: dict, path: Path = BUDGET_CSV) -> dict:
    """Append (or replace, by phase/case/seed) one run in the compute-budget CSV."""
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown compute-budget columns: {sorted(unknown)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = (str(row["phase"]), str(row["case"]), str(row["seed"]))
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if (r["phase"], r["case"], r["seed"]) != key]
    rows.append({c: _fmt(row.get(c)) for c in COLUMNS})
    rows.sort(key=lambda r: (r["phase"], r["case"], int(r["seed"] or 0)))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows[-1]


def read_budget(path: Path = BUDGET_CSV) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def run_row(phase, case, seed, result, conv, metrics, error_name, n_params, n_colloc, device):
    """Assemble a compute-budget row from a TrainResult and its convergence summary."""
    row = {
        "phase": phase,
        "case": case,
        "seed": seed,
        "error_name": error_name,
        "error": metrics[error_name],
        "adam_steps": result.adam_steps,
        "lbfgs_iters": result.lbfgs_iters,
        "time_adam_s": result.time_adam_s,
        "time_lbfgs_s": result.time_lbfgs_s,
        "time_total_s": result.time_total_s,
        "n_params": n_params,
        "n_collocation": n_colloc,
        "metrics_json": json.dumps({k: round(float(v), 8) for k, v in metrics.items()}),
        "notes": result.stopped,
    }
    row.update(conv)
    row.update(environment(device))
    return row


def save_checkpoint(model, path: Path, meta: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"config": model.config, "state": state, "meta": meta or {}}, path)
    return path


def load_checkpoint(path: Path, model_cls=None, device="cpu"):
    """Rebuild a model from a checkpoint (default class: :class:`pinncfd.nn.MLP`)."""
    from pinncfd.nn import MLP

    blob = torch.load(path, map_location=device, weights_only=True)
    model = (model_cls or MLP)(**blob["config"])
    model.load_state_dict(blob["state"])
    return model.to(device), blob["meta"]
