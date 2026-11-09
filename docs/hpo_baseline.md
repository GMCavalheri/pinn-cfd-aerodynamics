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
method starting from the hand-tuned genome (proxy validation error 0.588).

| Method | Candidates | Device time | Best validation error |
|---|---|---|---|
| NSGA-II | 24 (3 generations of 8) | 718 s | 0.133 |
| TPE + ASHA | 65 (60 pruned) | 602 s | 0.103 |
| Random | 21 | 606 s | 0.142 |

All three found the same kind of network: compressed coordinates, Fourier features, SiLU. One
search seed cannot rank them. Two things for the pre-registration:

- NSGA-II stops only at a generation boundary and overspent by 20 %. Either stop it inside a
  generation or size the generations so the overshoot is small, and report it either way.
- ASHA evaluated three times as many candidates as random search in the same time, which is
  its point; whether that buys accuracy needs the seeds.
