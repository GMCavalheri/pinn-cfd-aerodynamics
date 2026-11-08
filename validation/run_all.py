"""Rerun every phase's validation, in dependency order.

    uv run python validation/run_all.py               # full: retrains everything (~4-5 h GPU)
    uv run python validation/run_all.py --quick       # smoke test of every script (minutes)
    uv run python validation/run_all.py --plots-only  # figures and tables from saved results

Order matters: Phase 3 compares with the Phase 2 runs, Phase 4 times the better of the Phase 2
network and the Phase 3 champion. Full runs replace their rows in
``validation/results/compute_budget.csv``; ``--quick`` runs record nothing and write their
scratch output to ``validation/runs/``. The surrogate benchmark has no saved timings to
replot, so ``--plots-only`` skips it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    ("laplace_cylinder.py", "Phase 1: cylinder", True),
    ("airfoil_flowfield.py", "Phase 2: airfoil family", True),
    ("neuroevolution.py", "Phase 3: neuroevolution", True),
    ("surrogate_benchmark.py", "Phase 4: surrogate benchmark", False),
    ("euler_exploratory.py", "Phase 5: Euler (exploratory)", True),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    flags = ["--quick"] if args.quick else ["--plots-only"] if args.plots_only else []
    for name, title, replottable in SCRIPTS:
        if args.plots_only and not replottable:
            print(f"== {title}: skipped (no saved timings to replot)", flush=True)
            continue
        print(f"== {title}", flush=True)
        t0 = time.perf_counter()
        done = subprocess.run([sys.executable, str(HERE / name), *flags], cwd=HERE.parent)
        if done.returncode:
            print(f"   {name} failed (exit {done.returncode})", flush=True)
            return done.returncode
        print(f"   done in {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
