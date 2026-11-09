"""The RES-01 pilot scripts run end to end in their --quick mode (tiny budgets, CPU)."""

import importlib
import sys
from pathlib import Path

import pytest

RES01 = Path(__file__).resolve().parents[1] / "experiments" / "res01"


@pytest.mark.parametrize("script", ["pilot_joukowski", "pilot_airdata", "pilot_hpo"])
def test_pilot_quick_mode_runs(script, monkeypatch):
    monkeypatch.setenv("PINNCFD_DEVICE", "cpu")
    monkeypatch.syspath_prepend(str(RES01))
    mod = importlib.import_module(script)
    try:
        assert mod.main(["--quick"]) == 0
    finally:
        sys.modules.pop(script, None)
    out = RES01 / "runs"
    assert (out / f"{script}.md").exists() and (out / f"{script}.json").exists()
    assert (out / "compute_budget.csv").exists()
