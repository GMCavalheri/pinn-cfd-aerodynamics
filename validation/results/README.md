# Validation results

Written by the scripts in `validation/`; never edited by hand.

- `<phase>.md` — the tables each lesson quotes.
- `phase*_histories.json` — the evaluated error history of each run (for the convergence plots).
- `phase3_search.json` — every genome of the neuroevolution search with its fitness (also the
  resume checkpoint of the search).
- `checkpoints/` — small trained networks (≤ 100 kB each), reloaded by the notebooks and the
  surrogate benchmark without retraining.
- `compute_budget.csv` — one row per training run (and per classical-solver baseline), read by
  the compute-budget section of the paper. A rerun of the same `(phase, case, seed)` replaces
  its row. Baseline rows (`panel_*`) leave the training columns empty; their `time_total_s` is
  the solve time.

## `compute_budget.csv` columns

| Column | Meaning |
|---|---|
| `phase`, `case`, `seed` | Which run (the key) |
| `error_name`, `error` | The phase's headline error and its final value, on the validation split (Phase 1: relative L2 of the velocity on the field grid; Phase 2–3: relative L2 of the velocity over the validation airfoils; Phase 5: relative L2 of the primitive state) |
| `target_error` | The error the "steps to target" columns count to (1e-2 unless the phase says otherwise) |
| `steps_to_target` | First evaluated optimiser step with `error <= target_error`; empty if never reached. Steps are global: Adam steps 1…A, then L-BFGS iterations A+1…A+L |
| `steps_to_target_stage`, `steps_to_target_in_stage` | The same step as (optimiser, step within that optimiser) |
| `time_to_target_s` | Training wall time up to that step (evaluation time excluded) |
| `plateau_window`, `plateau_tol` | Plateau criterion: the running-best error improves by at most `plateau_tol` (relative) over any later window of `plateau_window` steps |
| `plateau_step`, `plateau_stage`, `plateau_in_stage`, `time_to_plateau_s` | First step from which that holds. **A plateau step equal to the last step means the error was still falling when the budget ran out** — no plateau was reached |
| `adam_steps`, `lbfgs_iters` | Iterations actually run (L-BFGS may stop early on its own tolerance) |
| `time_adam_s`, `time_lbfgs_s`, `time_total_s` | Training wall time per optimiser stage, CUDA-synchronised, evaluation excluded |
| `n_params`, `n_collocation` | Trainable parameters; collocation points per step (all loss terms) |
| `device`, `torch`, `cuda` | Hardware and library versions. The RTX 2060 runs clock-capped at 1000 MHz: GPU times are for that setting |
| `git_commit` | `HEAD` when the row was written, `-dirty` if tracked files differed from it (see below) |
| `date` | Date of the run |
| `metrics_json` | All metrics of the final evaluation, including, for training runs, the steps to the extra targets 1e-1, 3e-2, 1e-2, 1e-3 (`steps_to_1e-01` …; -1 if never reached) |
| `notes` | Why training stopped early, if it did |

## The `-dirty` flags

Up to Phase 3 the flag was set by *any* modified tracked file, and once this CSV was tracked
the runs modified it themselves, so it marks nearly every row.
No row was trained with uncommitted model or training code:

- **Phase 1** (`6ebfc08-dirty`): one uncommitted line of ruff configuration in
  `pyproject.toml` (the CSV was not tracked yet, and untracked files never counted).
- **Phase 2** (`55d74b0-dirty`): the CSV, and at most the table-wording and figure fixes to
  `validation/airfoil_flowfield.py` committed with the results (`1b4e245`); neither touches
  training.
- **Phase 3**: the first row (`search_000`, `410285c`) was written before the CSV changed. The
  rest read `41684f6-dirty`: the Lesson 2 commit landed during the search. That commit changed
  documentation only (the lesson and a docstring in `pinn_airfoil_flowfield/problem.py`), so
  every Phase 3 row ran the same code as `410285c`.

From Phase 4 on, `git_commit` ignores changes under `validation/results/`, so `-dirty` means
uncommitted code somewhere in the repository. Phase 4's rows are clean. Phase 5's rows
(`cf0cb28-dirty`) carry an uncommitted edit to `validation/surrogate_benchmark.py` (where its
`--quick` mode writes), made while the Euler runs trained; the Euler code and its script were
exactly as committed in `cf0cb28`.
