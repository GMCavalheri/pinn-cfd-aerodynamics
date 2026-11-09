"""RES-01 baselines for choosing a PINN configuration under a fixed compute budget.

* :func:`optuna_search.run_search` — random search, TPE, and either with ASHA pruning;
* :func:`nsga_budget.run_nsga` — the Phase 3 NSGA-II, generation by generation until the same
  budget is spent.

Both take an ``evaluate(genome, seed, on_eval) -> out`` callable (normally a partial of
``pinncfd.benchmark.run`` with ``test=False``) and a ``benchmark.ComputeLedger``.
"""
