# RES-01 experiments

Scripts and results for the RES-01 paper: neuroevolutionary PINNs for aerodynamic flow,
compared with hand tuning, random search, TPE and ASHA at an equal compute budget.

**Status: pilots only.** The scripts here are one-seed pilots that calibrate budgets and check
feasibility. The experimental design (cases, budgets, seeds, ablation arms, metrics) is
pre-registered before any final run; final scripts and results will be added here and marked as
such.

| Script | What it runs | Results |
|---|---|---|
| `pilot_joukowski.py` | case 2: the hand-tuned genome on plain, compressed and exact-map features at 0°, 4° and 8° | [`results/pilot_joukowski.md`](results/pilot_joukowski.md) |
| `pilot_airdata.py` | the inverse problem: the oracle's noise floor over every layout, noise level and incidence; the PINN at 8 taps | [`results/pilot_airdata.md`](results/pilot_airdata.md) |
| `pilot_hpo.py` | NSGA-II, TPE + ASHA and random search on case 2 at an equal 600 s budget (validation only) | [`results/pilot_hpo.md`](results/pilot_hpo.md) |

## What the pilots showed

- Case 2 on plain coordinates fails with the hand-tuned genome (velocity error 0.78–0.81),
  while the exact-map reference reaches 0.021–0.024. All three tuning methods found compressed
  coordinates with Fourier features (0.10–0.14 at the proxy budget).
- The inverse PINN is about ten times the oracle's noise floor in incidence, with the reference
  features.
- **Open issue:** the learned circulation scalar stays near zero in every run, and the
  exact-map network's lift is about 20 % low. See the
  [case 2 note](../../docs/pinn_joukowski.md#pilot-one-seed-hand-tuned-genome). To be fixed
  before the pre-registration.
- NSGA-II overspends its budget by up to a generation (20 % here).

Every run appends a row to [`results/compute_budget.csv`](results/compute_budget.csv) (phase
`res01-pilot`; columns as in [`validation/results/README.md`](../../validation/results/README.md),
with the genome and case parameters as JSON in `notes`). `--quick` runs a tiny CPU version of
any script into `runs/` (git-ignored); `tests/test_experiments.py` runs each quick mode.

```bash
uv run python experiments/res01/pilot_joukowski.py
```

Theory notes: [case 2](../../docs/pinn_joukowski.md),
[inverse air data](../../docs/inverse_airdata.md),
[tuning baselines](../../docs/hpo_baseline.md).
