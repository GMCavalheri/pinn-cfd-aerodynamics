"""The airfoil family and its train / validation / test split.

A case is one NACA 4-digit section at one angle of attack: (m, p, t, alpha). The family is
the box below. The PINN's physics loss is applied at parameters drawn anywhere in the box
(physics costs no data), while the panel-method *data* loss only sees the training cases;
validation cases steer the Phase 3 search and the test cases are touched once, at the end.

* training: 40 cases from a Latin hypercube over the box (seed 0);
* validation: 8 cases from an independent hypercube (seed 1);
* test: six named sections spread over the box, each at four angles none of which is a
  training angle by construction (they are fixed values, the hypercube's are continuous);
* extrapolation: cases outside the box, to show what happens beyond the training family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEG = np.pi / 180

# (low, high) for m, p, t, alpha [rad]
BOX = {
    "m": (0.0, 0.06),
    "p": (0.3, 0.5),
    "t": (0.08, 0.18),
    "alpha": (-4 * DEG, 8 * DEG),
}
LOW = np.array([v[0] for v in BOX.values()])
HIGH = np.array([v[1] for v in BOX.values()])


@dataclass(frozen=True)
class Case:
    m: float
    p: float
    t: float
    alpha: float  # rad

    @property
    def params(self) -> np.ndarray:
        return np.array([self.m, self.p, self.t, self.alpha])

    @property
    def name(self) -> str:
        p_digit = 0 if self.m == 0 else round(self.p * 10)  # p is irrelevant without camber
        code = f"{round(self.m * 100)}{p_digit}{round(self.t * 100):02d}"
        exact = np.allclose(
            [self.m, self.p, self.t],
            [round(self.m * 100) / 100, round(self.p * 10) / 10, round(self.t * 100) / 100],
        )
        shape = f"NACA {code}" if exact else f"m={self.m:.3f} p={self.p:.2f} t={self.t:.3f}"
        return f"{shape} α={self.alpha / DEG:+.1f}°"


def latin_hypercube(n: int, rng: np.random.Generator) -> np.ndarray:
    """n points in the box, one per stratum along every dimension (McKay et al. 1979)."""
    d = len(LOW)
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return LOW + u * (HIGH - LOW)


def uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    return LOW + rng.random((n, len(LOW))) * (HIGH - LOW)


def _cases(arr) -> list[Case]:
    return [Case(*map(float, row)) for row in arr]


def _naca(code: str, alpha_deg: float) -> Case:
    m, p = int(code[0]) / 100, int(code[1]) / 10
    # a symmetric section has no camber position; give it the box centre so it lies inside
    p = 0.4 if m == 0 else p
    return Case(m, p, int(code[2:]) / 100, alpha_deg * DEG)


TEST_SECTIONS = ("0012", "1408", "2412", "3315", "4415", "5518")
TEST_ALPHAS_DEG = (-3.0, 1.0, 4.0, 7.0)
EXTRAPOLATION = (("0021", 4.0), ("2412", 12.0), ("8412", 4.0), ("2406", 4.0))


def train_cases(n: int = 40) -> list[Case]:
    return _cases(latin_hypercube(n, np.random.default_rng(0)))


def val_cases(n: int = 8) -> list[Case]:
    return _cases(latin_hypercube(n, np.random.default_rng(1)))


def test_cases() -> list[Case]:
    return [_naca(s, a) for s in TEST_SECTIONS for a in TEST_ALPHAS_DEG]


def extrapolation_cases() -> list[Case]:
    return [_naca(s, a) for s, a in EXTRAPOLATION]
