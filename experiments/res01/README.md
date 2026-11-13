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

- **Case 2 with the exact-map reference is solved.** The network learns three basis flows and
  the circulation follows from the Kutta condition. Velocity error 0.004–0.008, and the
  circulation within 2 % of exact at 0°, 4° and 8°. See the
  [case 2 note](../../docs/pinn_joukowski.md#pilot-one-seed-hand-tuned-genome).
- **On plain or compressed coordinates case 2 still fails** (0.80–0.82 and 0.71, circulation
  about zero): the basis flows are singular at the cusp. This is a finding for the paper,
  not a bug. All three tuning methods chose compressed coordinates with Fourier features
  (0.19–0.21 at the proxy budget).
- **The inverse PINN regressed** to a 4.2–4.5° incidence error with the reference features
  (0.16–0.32° before): the tap misfit bends the basis flows instead of moving the incidence.
  See the [air-data note](../../docs/inverse_airdata.md#pilot-one-seed-8-taps-4). A fix is
  proposed there and waits for a decision.
- NSGA-II overspends its budget by up to a generation (47 % here).

Total training time of the current pilots: 4.6 GPU-hours on the RTX 2060 (2.8 for case 2, 1.2
for air data, 0.6 for the tuning pilot).

Every run appends a row to [`results/compute_budget.csv`](results/compute_budget.csv) (phase
`res01-pilot`; columns as in [`validation/results/README.md`](../../validation/results/README.md),
with the genome and case parameters as JSON in `notes`). A rerun replaces the rows with the same
phase, case and seed, so the record holds the latest run of each. Tuning-pilot rows with
`git_commit` `4144730` are left from the first pilot (before the basis flows), numbered past
the candidates of the current one. `--quick` runs a tiny CPU version of
any script into `runs/` (git-ignored); `tests/test_experiments.py` runs each quick mode.

```bash
uv run python experiments/res01/pilot_joukowski.py
```

Theory notes: [case 2](../../docs/pinn_joukowski.md),
[inverse air data](../../docs/inverse_airdata.md),
[tuning baselines](../../docs/hpo_baseline.md).
