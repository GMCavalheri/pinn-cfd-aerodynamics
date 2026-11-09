"""RES-01 pilot: tuning methods on case 2 at a small equal budget.

On the Joukowski section at 4 degrees (genes: ``JOUKOWSKI_SPACE``), three methods each get
the same device-time budget, every candidate trained with the Phase 3 proxy budget and scored
on the validation set only:

* NSGA-II (population 8, the Phase 3 search) run generation by generation;
* TPE with ASHA pruning;
* random search.

All three start from the hand-tuned genome. One search seed each. The best genome of each is
reported with its proxy validation error; retraining winners at the full budget, seeds and the
test set belong to the pre-registered experiment, not this pilot.

Writes ``results/pilot_hpo.md`` and ``.json`` and appends every trial to
``results/compute_budget.csv``. ``--quick`` runs a tiny version into ``runs/``.
"""

from __future__ import annotations

import math
from functools import partial

from pinncfd import benchmark as B
from pinncfd.hpo.nsga_budget import run_nsga
from pinncfd.hpo.optuna_search import run_search
from pinncfd.neuroevolution_search import genome as G

from pilot_common import PHASE, g, setup, table, write_json

BUDGET_S = 600.0  # per method
PROXY = dict(adam_steps=400, lbfgs_iters=100, eval_every=100, resample_every=100)
QUICK_PROXY = dict(adam_steps=10, lbfgs_iters=2, eval_every=5, resample_every=5)
ALPHA = math.radians(4.0)


def main(argv=None) -> int:
    args, out_dir, budget_csv, device = setup(__doc__.splitlines()[0], argv)
    proxy = QUICK_PROXY if args.quick else PROXY
    budget_s = 3.0 if args.quick else BUDGET_S
    kw = dict(alpha=ALPHA) | (dict(n_eval=64, n_eval_body=50) if args.quick else {})
    space = G.JOUKOWSKI_SPACE
    hand = G.JOUKOWSKI_HAND_GENOME
    if args.quick:
        space = space.freeze({"n_interior": 1024})
        hand = dict(hand, n_interior=1024)
    counter = {"n": 0}

    def evaluate(method, genome, seed, on_eval):
        out = B.run("joukowski", genome, proxy, seed, device, kw, on_eval=on_eval, test=False)
        B.record(PHASE, f"hpo_{method}_{counter['n']:03d}", out, device, budget_csv)
        counter["n"] += 1
        return out

    results = {}
    ck = out_dir / "pilot_hpo_nsga.json"
    ck.unlink(missing_ok=True)
    counter["n"] = 0
    results["nsga"] = run_nsga(
        partial(evaluate, "nsga"),
        space,
        B.ComputeLedger(budget_s),
        population_size=2 if args.quick else 8,
        seed=0,
        checkpoint=ck,
        initial=[hand],
    )
    for name, sampler, pruner in (("tpe_asha", "tpe", "asha"), ("random", "random", None)):
        counter["n"] = 0
        results[name] = run_search(
            partial(evaluate, name),
            space,
            B.ComputeLedger(budget_s),
            sampler=sampler,
            pruner=pruner,
            seed=0,
            initial=[hand],
        )
    rows = []
    for name, r in results.items():
        n_runs = r["ledger"]["runs"]
        hand_err = (
            r["evaluated"][0]["fitness"]["error"] if name == "nsga" else r["trials"][0]["error"]
        )
        rows.append(
            [
                name,
                n_runs,
                g(r["ledger"]["spent_s"], 4),
                g(hand_err),
                g(r["best_error"]),
                G.describe(r["best_genome"]) if r["best_genome"] else "—",
            ]
        )
    md = [
        "# RES-01 pilot: tuning methods on case 2 at an equal budget",
        "",
        f"Joukowski section at 4°, gene space `JOUKOWSKI_SPACE`, {budget_s:g} s of device time "
        f"per method (training + evaluation), proxy budget {proxy['adam_steps']} Adam + "
        f"{proxy['lbfgs_iters']} L-BFGS per candidate, validation set only. Each method's first "
        "candidate is the hand-tuned genome. One search seed: a pilot, not a result.",
        "",
        table(
            [
                "method",
                "candidates",
                "spent [s]",
                "hand-tuned (proxy) val error",
                "best val error",
                "best genome",
            ],
            rows,
        ),
        "",
    ]
    (out_dir / "pilot_hpo.md").write_text("\n".join(md))
    slim = {
        k: {kk: vv for kk, vv in v.items() if kk != "evaluated"}
        | {"n_evaluated": len(v.get("evaluated", []))}
        for k, v in results.items()
    }
    write_json(out_dir / "pilot_hpo.json", slim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
