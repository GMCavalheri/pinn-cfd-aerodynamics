"""RES-01 pilot: the inverse air-data problem, oracle noise floor and the PINN.

1. The model-based oracle (least squares on the exact pressure) over every layout (4, 6, 8,
   10 taps), noise level (0, 1, 2, 5 % of the dynamic pressure) and incidence (0, 4, 8
   degrees), 20 validation noise draws each: the noise floor any estimator faces.
2. The air-data PINN with 8 taps at 4 degrees and U = 1.15 (starting from 0 degrees and
   U = 1), noise 0 and 2 %, validation draw 0, on plain coordinates and on the exact map
   (reference). One seed each.

Writes ``results/pilot_airdata.md`` and ``.json`` and appends to
``results/compute_budget.csv``. ``--quick`` runs a tiny version into ``runs/``.
"""

from __future__ import annotations

import math

import numpy as np

from pinncfd import benchmark as B
from pinncfd.inverse_airdata import oracle, taps
from pinncfd.neuroevolution_search import genome as G
from pinncfd.pinn_joukowski import JoukowskiFlow

from pilot_common import PHASE, g, history, setup, table, write_json

U_TRUE = 1.15
BUDGET = dict(adam_steps=3000, lbfgs_iters=500, eval_every=250, resample_every=250)
BUDGET |= dict(plateau_window=1000)
QUICK = dict(adam_steps=20, lbfgs_iters=5, eval_every=10, resample_every=10)
PINN_RUNS = [("plain", 0.0), ("plain", 0.02), ("exact_map", 0.0), ("exact_map", 0.02)]


def oracle_grid(draws: int) -> list[dict]:
    flow = JoukowskiFlow()
    out = []
    for n in taps.LAYOUTS:
        for level in taps.NOISE_LEVELS:
            da, du = [], []
            for a_deg in (0.0, 4.0, 8.0):
                a = math.radians(a_deg)
                for d in range(draws):
                    dp = taps.readings(flow, n, a, U_TRUE, level, "val", d)
                    ah, uh = oracle.fit(flow, n, dp)
                    da.append(math.degrees(abs(ah - a)))
                    du.append(abs(uh - U_TRUE) / U_TRUE)
            out.append(
                {
                    "taps": n,
                    "noise": level,
                    "alpha_err_mean_deg": float(np.mean(da)),
                    "alpha_err_p95_deg": float(np.percentile(da, 95)),
                    "u_err_mean": float(np.mean(du)),
                }
            )
    return out


def main(argv=None) -> int:
    args, out_dir, budget_csv, device = setup(__doc__.splitlines()[0], argv)
    grid = oracle_grid(2 if args.quick else 20)
    budget = QUICK if args.quick else BUDGET
    small = dict(n_interior=128, n_eval=128, n_eval_body=100) if args.quick else {}
    flow = JoukowskiFlow()
    pinn_rows, record = [], []
    for features, level in PINN_RUNS[:1] if args.quick else PINN_RUNS:
        genome = dict(G.AIRDATA_HAND_GENOME, features=features)
        kw = dict(alpha=math.radians(4.0), u=U_TRUE, n_taps=8, noise=level, draw=0) | small
        out = B.run("airdata", genome, budget, 0, device, kw)
        name = f"airdata_{features}_n8_noise{level:g}"
        B.record(PHASE, name, out, device, budget_csv)
        dp = taps.readings(flow, 8, math.radians(4.0), U_TRUE, level, "val", 0)
        a_or, u_or = oracle.fit(flow, 8, dp)
        val = out["val"]
        print(f"{name}: alpha err {val['error']:.3g} deg, U err {val.get('u_err', math.inf):.3g}")
        pinn_rows.append(
            [
                features,
                f"{100 * level:g} %",
                g(val["error"]),
                g(val.get("u_err")),
                g(math.degrees(abs(a_or - math.radians(4.0)))),
                g(abs(u_or - U_TRUE) / U_TRUE),
                g(val.get("field_error")),
                g(out["time_train_s"], 3),
            ]
        )
        record.append(
            {
                "name": name,
                "features": features,
                "noise": level,
                "val": val,
                "scalars": out["scalars"],
                "oracle": {"alpha": a_or, "u": u_or},
                "time_train_s": out["time_train_s"],
                "history": history(out),
            }
        )
    grid_rows = [
        [
            r["taps"],
            f"{100 * r['noise']:g} %",
            g(r["alpha_err_mean_deg"]),
            g(r["alpha_err_p95_deg"]),
            g(r["u_err_mean"]),
        ]
        for r in grid
    ]
    md = [
        "# RES-01 pilot: inverse air data",
        "",
        "## Oracle noise floor",
        "",
        "Least squares on the exact pressure model; 20 validation noise draws at each of 0°, 4° "
        f"and 8°, U = {U_TRUE}. Errors are absolute for α, relative for U.",
        "",
        table(
            ["taps", "noise", "α error, mean [°]", "α error, 95th pct [°]", "U error, mean"],
            grid_rows,
        ),
        "",
        "## PINN (one seed, validation draw 0)",
        "",
        f"8 taps, true α = 4° and U = {U_TRUE}, started from 0° and U = 1. Budget "
        f"{budget['adam_steps']} Adam steps + {budget['lbfgs_iters']} L-BFGS iterations. "
        "`exact_map` is the reference features (never offered to a search).",
        "",
        table(
            [
                "features",
                "noise",
                "α error [°]",
                "U error",
                "oracle α error [°]",
                "oracle U error",
                "field velocity error",
                "training time [s]",
            ],
            pinn_rows,
        ),
        "",
    ]
    (out_dir / "pilot_airdata.md").write_text("\n".join(md))
    write_json(out_dir / "pilot_airdata.json", {"oracle_grid": grid, "pinn": record})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
