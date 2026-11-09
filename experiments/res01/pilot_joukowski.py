"""RES-01 pilot: case 2 (Joukowski section) with the hand-tuned genome, by input features.

Runs the hand-tuned network (``JOUKOWSKI_HAND_GENOME``) on plain coordinates at 0, 4 and 8
degrees, on compressed coordinates at 4 degrees, and on the exact circle-plane map (the
"answer for free" reference, never offered to a search) at all three angles. One seed each.

Writes ``results/pilot_joukowski.md`` and ``.json`` and appends to
``results/compute_budget.csv``. ``--quick`` runs a tiny version into ``runs/``.
"""

from __future__ import annotations

import math

from pinncfd import benchmark as B
from pinncfd.neuroevolution_search import genome as G

from pilot_common import PHASE, g, history, setup, table, write_json

BUDGET = dict(adam_steps=3000, lbfgs_iters=500, eval_every=250, resample_every=250)
BUDGET |= dict(plateau_window=1000)
QUICK = dict(adam_steps=20, lbfgs_iters=5, eval_every=10, resample_every=10)
RUNS = [
    ("plain", 0.0),
    ("plain", 4.0),
    ("plain", 8.0),
    ("compress", 4.0),
    ("exact_map", 0.0),
    ("exact_map", 4.0),
    ("exact_map", 8.0),
]


def main(argv=None) -> int:
    args, out_dir, budget_csv, device = setup(__doc__.splitlines()[0], argv)
    budget = QUICK if args.quick else BUDGET
    small = dict(n_interior=128, n_eval=128, n_eval_body=100) if args.quick else {}
    runs = RUNS[:2] if args.quick else RUNS
    rows, record = [], []
    for features, a_deg in runs:
        genome = dict(G.JOUKOWSKI_HAND_GENOME, features=features)
        kw = dict(alpha=math.radians(a_deg)) | small
        out = B.run("joukowski", genome, budget, 0, device, kw)
        name = f"joukowski_{features}_a{a_deg:g}"
        B.record(PHASE, name, out, device, budget_csv)
        val, test = out["val"], out["test"] or {}
        print(f"{name}: val {val['error']:.4g}  test {test.get('error', math.inf):.4g}")
        rows.append(
            [
                features,
                f"{a_deg:g}°",
                g(val["error"]),
                g(test.get("error")),
                g(test.get("cp_rms_mid")),
                g(test.get("cl_err_circulation")),
                g(test.get("cl_err_pressure")),
                g(out["time_train_s"], 3),
            ]
        )
        record.append(
            {
                "name": name,
                "features": features,
                "alpha_deg": a_deg,
                "val": val,
                "test": test,
                "scalars": out["scalars"],
                "time_train_s": out["time_train_s"],
                "history": history(out),
            }
        )
    header = [
        "features",
        "α",
        "val velocity error",
        "test velocity error",
        "test Cp RMS (5–95 % chord)",
        "test cl error (circulation)",
        "test cl error (pressure)",
        "training time [s]",
    ]
    md = [
        "# RES-01 pilot: case 2, hand-tuned genome by input features",
        "",
        "One seed (0) per row; budget "
        f"{budget['adam_steps']} Adam steps + {budget['lbfgs_iters']} L-BFGS iterations. "
        "`exact_map` is the reference that hands the network the solution's own coordinates; "
        "it is never offered to a search. A pilot, not a result: it calibrates the "
        "pre-registration.",
        "",
        table(header, rows),
        "",
    ]
    (out_dir / "pilot_joukowski.md").write_text("\n".join(md))
    write_json(out_dir / "pilot_joukowski.json", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
