# RES-01 baselines: tuning methods at an equal compute budget

Code: [`src/pinncfd/hpo/`](../src/pinncfd/hpo/),
[`src/pinncfd/benchmark.py`](../src/pinncfd/benchmark.py) ·
Pilot: [`experiments/res01/results/pilot_hpo.md`](../experiments/res01/results/pilot_hpo.md) ·
Tests: [`test_hpo.py`](../tests/test_hpo.py), [`test_benchmark.py`](../tests/test_benchmark.py)

Phase 3 found that NSGA-II beat the hand-tuned network, but the evolved network also had a
40-minute search the hand-tuned one did not. RES-01 asks the fairer question: given the same
device time, does neuroevolution beat the standard ways of tuning a network?

## One entry point per case

`benchmark.run(case, genome, budget, seed, device)` builds, trains and scores the cylinder, the
Joukowski section, the NACA family or the air-data problem from a genome, and returns the
validation metrics, the test metrics (unless a search asks it not to: a search never sees the
test set), the time spent and the learned scalars. `benchmark.record` writes a row to RES-01's
own compute record, `experiments/res01/results/compute_budget.csv`, with the genome in its
notes.

## Equal compute

`ComputeLedger` charges each method the training and evaluation time of every run it makes,
search included. A method stops after the run (or, for NSGA-II, the generation) that exhausts
its budget, so the overshoot is at most one run or one generation and is reported.

## The methods

| Method | How it proposes candidates | Code |
|---|---|---|
| Hand-tuned | the Phase 2 configuration | `genome.JOUKOWSKI_HAND_GENOME` |
| NSGA-II | Phase 3's search (accuracy and size), a generation at a time | `hpo.nsga_budget.run_nsga` |
| Random search | independent draws from the gene space (Bergstra & Bengio 2012) | `hpo.optuna_search`, `sampler="random"` |
| TPE | tree-structured Parzen estimator (Bergstra et al. 2011) | `sampler="tpe"` |
| ASHA | successive halving on the validation curve (Li et al. 2020), with random or TPE | `pruner="asha"` |

All search the same `GeneSpace`. ASHA reads each candidate's validation error at every
evaluation through the trainer's `on_eval` hook, and a pruned candidate is charged for the
time it used.

## Ablation arms

`genome.ablation_space(space, hand, groups)` freezes every gene outside the chosen groups at
its hand-tuned value. The groups are architecture (depth, width, activation, Fourier features,
input features), loss weighting and sampling (every loss weight and the collocation count) and
the optimiser (learning rate). Which groups each arm evolves is fixed in the pre-registration.

## Pilot (one search seed, case 2 at 4°)

[`experiments/res01/results/pilot_hpo.md`](../experiments/res01/results/pilot_hpo.md):
600 s per method, proxy budget 400 Adam steps and 100 L-BFGS iterations per candidate, every
method starting from the hand-tuned genome (proxy validation error 0.37). Rerun after case 2
moved to superposed basis flows (no Kutta weight gene); each candidate now costs about four
times as much as before, so far fewer fit in the budget.

| Method | Candidates | Device time | Best validation error |
|---|---|---|---|
| NSGA-II | 8 (1 generation) | 879 s | 0.208 |
| TPE + ASHA | 14 (11 pruned) | 619 s | 0.186 |
| Random | 6 | 626 s | 0.186 |

All three again chose compressed coordinates with Fourier features. TPE and random search
found the same best genome: TPE starts with random draws from the same seed, and 14 candidates
end inside that start-up phase. One search seed cannot rank the methods. For the
pre-registration:

- NSGA-II stops only at a generation boundary and overspent by 47 % (20 % before, with cheaper
  candidates). Either stop it inside a generation or size the generations so the overshoot is
  small, and report it either way.
- The budget must fit enough candidates for TPE to leave its start-up phase, at least tens.
- The hand-tuned genome scores 0.37 at the proxy budget but 0.81 at the full one (the
  [case 2 pilot](pinn_joukowski.md#pilot-one-seed-hand-tuned-genome)): on these inputs, longer
  training makes the network worse, so proxy errors can mislead. Retrain the winners at the full
  budget before comparing them.
- ASHA evaluated more candidates than random search in the same time, which is its point;
  whether that buys accuracy needs the seeds.
