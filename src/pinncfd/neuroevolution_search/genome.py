"""The genome: what neuroevolution searches over for the airfoil-family PINN.

As in the RUL project (``src/rul/evolution/genome.py``, commit addc209), a genome is a flat,
fixed-length set of independent genes, each with its own domain and sampling rule: a
categorical choice, a uniform range, or a log-uniform range. Uniform crossover and per-gene
resampling mutation then need no special cases.

The genes cover the three things a PINN practitioner otherwise tunes by hand:

* **architecture** — depth, width, activation, optional Fourier features and their scale;
* **optimisation** — the Adam learning rate and the number of interior collocation points
  per shape;
* **loss weighting** — the weight of every loss term relative to the PDE residual (whose
  weight stays 1, since only ratios matter). This is the thesis's core contribution:
  balancing data and physics terms by evolution instead of by hand.
"""

from __future__ import annotations

import numpy as np

CHOICE = {
    "depth": [3, 4, 5, 6],
    "width": [32, 48, 64, 96],
    "activation": ["tanh", "sin", "silu"],
    "fourier_features": [0, 16, 32],
    "n_interior": [192, 384, 768],
}
UNIFORM = {
    "fourier_scale": (0.25, 2.0),
}
LOG_UNIFORM = {
    "lr": (3e-4, 3e-3),
    "w_body": (0.1, 100.0),
    "w_tangency": (0.1, 100.0),
    "w_kutta": (0.1, 100.0),
    "w_far": (0.1, 100.0),
    "w_data": (0.1, 100.0),
}
GENES = (*CHOICE, *UNIFORM, *LOG_UNIFORM)

# The hand-tuned Phase 2 configuration, expressed as a genome (seeded into generation 0 so the
# search starts no worse than the manual choice).
PHASE2_GENOME = {
    "depth": 5,
    "width": 64,
    "activation": "tanh",
    "fourier_features": 0,
    "n_interior": 384,
    "fourier_scale": 1.0,
    "lr": 1e-3,
    "w_body": 10.0,
    "w_tangency": 1.0,
    "w_kutta": 10.0,
    "w_far": 10.0,
    "w_data": 10.0,
}


def _sample_gene(name: str, rng: np.random.Generator):
    if name in CHOICE:
        choices = CHOICE[name]
        return choices[rng.integers(len(choices))]
    if name in UNIFORM:
        return float(rng.uniform(*UNIFORM[name]))
    lo, hi = LOG_UNIFORM[name]
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_genome(rng: np.random.Generator) -> dict:
    return {name: _sample_gene(name, rng) for name in GENES}


def mutate(genome: dict, rng: np.random.Generator, rate: float = 0.2) -> dict:
    """Independently resample each gene with probability ``rate``."""
    return {k: (_sample_gene(k, rng) if rng.random() < rate else v) for k, v in genome.items()}


def crossover(a: dict, b: dict, rng: np.random.Generator) -> dict:
    """Uniform crossover: each gene independently comes from one parent or the other."""
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in GENES}


def in_domain(genome: dict) -> bool:
    for k, v in genome.items():
        if k in CHOICE and v not in CHOICE[k]:
            return False
        if k in UNIFORM and not UNIFORM[k][0] <= v <= UNIFORM[k][1]:
            return False
        if k in LOG_UNIFORM and not LOG_UNIFORM[k][0] <= v <= LOG_UNIFORM[k][1]:
            return False
    return set(genome) == set(GENES)


def describe(genome: dict) -> str:
    ff = (
        f", FF {genome['fourier_features']}@{genome['fourier_scale']:.2f}"
        if genome["fourier_features"]
        else ""
    )
    return (
        f"{genome['depth']}×{genome['width']} {genome['activation']}{ff}, "
        f"lr {genome['lr']:.1e}, N {genome['n_interior']}, "
        f"w body/tan/kutta/far/data {genome['w_body']:.3g}/{genome['w_tangency']:.3g}/"
        f"{genome['w_kutta']:.3g}/{genome['w_far']:.3g}/{genome['w_data']:.3g}"
    )
