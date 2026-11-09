"""Helpers shared by the RES-01 pilot scripts.

Pilots are one-seed runs that calibrate budgets and check feasibility before the experimental
design is pre-registered (RES-01 weeks 3-5). They are not the paper's results. Every row they
write to the compute record has the phase ``res01-pilot``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pinncfd.device import get_device

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SCRATCH = HERE / "runs"
PHASE = "res01-pilot"


def setup(description: str, argv=None):
    """Parse ``--quick``; return (args, output directory, compute-record path, device)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--quick", action="store_true", help="tiny CPU-sized smoke run into runs/ (git-ignored)"
    )
    args = parser.parse_args(argv)
    out = SCRATCH if args.quick else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    return args, out, out / "compute_budget.csv", get_device()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n")


def g(x, digits: int = 3) -> str:
    """A number for a Markdown table: 3 significant digits, '—' for None, '∞' for inf."""
    if x is None:
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:.{digits}g}"


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def history(out: dict) -> list[dict]:
    res = out.get("result")
    if res is None:
        return []
    keep = ("step", "stage", "loss", "error", "time_s")
    return [{k: h[k] for k in keep if k in h} for h in res.history]
