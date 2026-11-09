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

A :class:`GeneSpace` holds the domains. ``PHASE3_SPACE`` is the Phase 3 space and the default
everywhere, with the same random-number stream as before the class existed, so the Phase 3
search reproduces. RES-01 adds two things on top:

* ``JOUKOWSKI_SPACE`` for the single-section Joukowski case: no data weight (that case is
  physics only) and an input-feature gene, ``features``;
* **frozen genes** (:meth:`GeneSpace.freeze`): held at a fixed value, never sampled, mutated or
  crossed. Freezing one group of genes at their hand-tuned values gives the ablations
  (architecture only, loss weighting and sampling only, or both); see :func:`ablation_space`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GeneSpace:
    choice: dict
    uniform: dict
    log_uniform: dict
    fixed: dict = field(default_factory=dict)

    @property
    def genes(self) -> tuple[str, ...]:
        return (*self.choice, *self.uniform, *self.log_uniform)

    @property
    def free(self) -> tuple[str, ...]:
        return tuple(g for g in self.genes if g not in self.fixed)

    def sample_gene(self, name: str, rng: np.random.Generator):
        if name in self.fixed:
            return self.fixed[name]
        if name in self.choice:
            choices = self.choice[name]
            return choices[rng.integers(len(choices))]
        if name in self.uniform:
            return float(rng.uniform(*self.uniform[name]))
        lo, hi = self.log_uniform[name]
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    def sample(self, rng: np.random.Generator) -> dict:
        return {name: self.sample_gene(name, rng) for name in self.genes}

    def mutate(self, genome: dict, rng: np.random.Generator, rate: float = 0.2) -> dict:
        """Independently resample each free gene with probability ``rate``."""
        return {
            k: (v if k in self.fixed else self.sample_gene(k, rng) if rng.random() < rate else v)
            for k, v in genome.items()
        }

    def crossover(self, a: dict, b: dict, rng: np.random.Generator) -> dict:
        """Uniform crossover: each free gene independently comes from one parent or the other."""
        return {
            k: (self.fixed[k] if k in self.fixed else a[k] if rng.random() < 0.5 else b[k])
            for k in self.genes
        }

    def in_domain(self, genome: dict) -> bool:
        for k, v in genome.items():
            if k in self.fixed and v != self.fixed[k]:
                return False
            if k in self.choice and v not in self.choice[k]:
                return False
            if k in self.uniform and not self.uniform[k][0] <= v <= self.uniform[k][1]:
                return False
            if k in self.log_uniform and not self.log_uniform[k][0] <= v <= self.log_uniform[k][1]:
                return False
        return set(genome) == set(self.genes)

    def freeze(self, values: dict) -> GeneSpace:
        """A copy with the genes in ``values`` held at those values (each must be in its domain)."""
        unknown = set(values) - set(self.genes)
        if unknown:
            raise KeyError(f"not genes of this space: {sorted(unknown)}")
        out = GeneSpace(self.choice, self.uniform, self.log_uniform, {**self.fixed, **values})
        probe = {g: out.fixed.get(g) for g in out.genes}
        for g in out.free:
            probe[g] = out.sample_gene(g, np.random.default_rng(0))
        if not GeneSpace(self.choice, self.uniform, self.log_uniform).in_domain(probe):
            raise ValueError(f"frozen values outside their domains: {values}")
        return out

    def without(self, *names: str) -> GeneSpace:
        """A copy without the genes ``names``."""

        def drop(d):
            return {k: v for k, v in d.items() if k not in names}

        return GeneSpace(
            drop(self.choice), drop(self.uniform), drop(self.log_uniform), drop(self.fixed)
        )

    def with_choice(self, name: str, options: list) -> GeneSpace:
        """A copy with an extra categorical gene, placed after the existing categorical ones."""
        return GeneSpace(
            {**self.choice, name: list(options)}, self.uniform, self.log_uniform, self.fixed
        )


PHASE3_SPACE = GeneSpace(
    choice={
        "depth": [3, 4, 5, 6],
        "width": [32, 48, 64, 96],
        "activation": ["tanh", "sin", "silu"],
        "fourier_features": [0, 16, 32],
        "n_interior": [192, 384, 768],
    },
    uniform={
        "fourier_scale": (0.25, 2.0),
    },
    log_uniform={
        "lr": (3e-4, 3e-3),
        "w_body": (0.1, 100.0),
        "w_tangency": (0.1, 100.0),
        "w_kutta": (0.1, 100.0),
        "w_far": (0.1, 100.0),
        "w_data": (0.1, 100.0),
    },
)

# Module-level names of the Phase 3 space, as before GeneSpace existed.
CHOICE = PHASE3_SPACE.choice
UNIFORM = PHASE3_SPACE.uniform
LOG_UNIFORM = PHASE3_SPACE.log_uniform
GENES = PHASE3_SPACE.genes

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

# RES-01 case 2: one Joukowski section, physics only. The feature gene chooses how the network
# sees (x, y): affine-scaled ("plain") or radially compressed ("compress"). The exact inverse
# Joukowski map is deliberately not an option: on a Joukowski section it is the solution's own
# coordinate system, so it would hand the network the answer.
# The interior-point counts of the family are per shape, with 32 shapes in every batch; a single
# section gets its own counts, with the hand-tuned value 2048 in the middle.
FEATURES = ["plain", "compress"]
JOUKOWSKI_SPACE = (
    PHASE3_SPACE.without("w_data")
    .with_choice("n_interior", [1024, 2048, 4096])
    .with_choice("features", FEATURES)
)
JOUKOWSKI_HAND_GENOME = {
    **{k: v for k, v in PHASE2_GENOME.items() if k != "w_data"},
    "n_interior": 2048,
    "features": "plain",
}

# Gene groups for the RES-01 ablation. The pre-registration fixes which groups each arm evolves.
GROUPS = {
    "architecture": (
        "depth",
        "width",
        "activation",
        "fourier_features",
        "fourier_scale",
        "features",
    ),
    "loss_and_sampling": ("w_body", "w_tangency", "w_kutta", "w_far", "w_data", "n_interior"),
    "optimiser": ("lr",),
}


def ablation_space(space: GeneSpace, hand: dict, free_groups) -> GeneSpace:
    """``space`` with every gene outside ``free_groups`` frozen at its ``hand`` value."""
    free = {g for name in free_groups for g in GROUPS[name]}
    return space.freeze({g: hand[g] for g in space.genes if g not in free})


def sample_genome(rng: np.random.Generator, space: GeneSpace = PHASE3_SPACE) -> dict:
    return space.sample(rng)


def mutate(
    genome: dict, rng: np.random.Generator, rate: float = 0.2, space: GeneSpace = PHASE3_SPACE
) -> dict:
    """Independently resample each gene with probability ``rate``."""
    return space.mutate(genome, rng, rate)


def crossover(a: dict, b: dict, rng: np.random.Generator, space: GeneSpace = PHASE3_SPACE) -> dict:
    """Uniform crossover: each gene independently comes from one parent or the other."""
    return space.crossover(a, b, rng)


def in_domain(genome: dict, space: GeneSpace = PHASE3_SPACE) -> bool:
    return space.in_domain(genome)


_WEIGHT_LABELS = {
    "w_body": "body",
    "w_tangency": "tan",
    "w_kutta": "kutta",
    "w_far": "far",
    "w_data": "data",
}


def describe(genome: dict) -> str:
    ff = (
        f", FF {genome['fourier_features']}@{genome['fourier_scale']:.2f}"
        if genome["fourier_features"]
        else ""
    )
    feat = f", {genome['features']} features" if "features" in genome else ""
    ws = [k for k in _WEIGHT_LABELS if k in genome]
    labels = "/".join(_WEIGHT_LABELS[k] for k in ws)
    values = "/".join(f"{genome[k]:.3g}" for k in ws)
    return (
        f"{genome['depth']}×{genome['width']} {genome['activation']}{ff}{feat}, "
        f"lr {genome['lr']:.1e}, N {genome['n_interior']}, "
        f"w {labels} {values}"
    )
