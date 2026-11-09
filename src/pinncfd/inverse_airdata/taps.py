"""Surface pressure taps on the Joukowski section and their synthetic readings.

A flush air data system (FADS) infers the flight condition from pressures measured at ports
flush with the skin around the nose, where the pressure field is most sensitive to incidence
and speed. Here the "aircraft" is the Joukowski section of RES-01 case 2, the readings are the
exact potential-flow pressures, and the unknowns are the incidence alpha and the freestream
speed U (density and freestream static pressure known).

Layouts: ``n`` taps symmetric about the leading edge in the circle-plane angle, spanning
+-TAP_SPAN about it, so they cluster where the nose is curved. Readings, with rho = 1:

    dp = p - p_inf = (1/2) (U^2 - |V|^2) = (1/2) U^2 Cp(alpha),

plus independent Gaussian noise of standard deviation ``level`` x (1/2) U^2 (a fraction of the
freestream dynamic pressure). Noise draws come from fixed seeds, disjoint between the
validation draws a search may use and the test draws used for reporting.
"""

from __future__ import annotations

import numpy as np

from pinncfd.pinn_joukowski.exact import JoukowskiFlow

LAYOUTS = (4, 6, 8, 10)
NOISE_LEVELS = (0.0, 0.01, 0.02, 0.05)
TAP_SPAN = np.radians(60.0)  # half-span of the layout, in circle-plane angle about the LE
SPLIT_BASE = {"val": 10_000, "test": 20_000}


def tap_angles(n: int) -> np.ndarray:
    """Circle-plane angle offsets from the leading edge, symmetric, outermost at +-TAP_SPAN."""
    return np.linspace(-TAP_SPAN, TAP_SPAN, n)


def tap_points(flow: JoukowskiFlow, n: int) -> np.ndarray:
    """(n, 2) normalised tap positions on the body, lower surface first."""
    te = np.angle(flow.b - flow.zeta0)
    zeta = flow.zeta0 + flow.a * np.exp(1j * (te + np.pi + tap_angles(n)))
    x, y = flow.from_circle(zeta)
    return np.column_stack([x, y])


def exact_cp(flow: JoukowskiFlow, n: int, alpha: float) -> np.ndarray:
    """Exact Cp at the taps, from the circle plane (no inversion of the map needed)."""
    te = np.angle(flow.b - flow.zeta0)
    th = te + np.pi + tap_angles(n)
    zp = flow.a * np.exp(1j * th)
    zeta = flow.zeta0 + zp
    g = flow.foil.circulation(alpha)
    d = np.exp(-1j * alpha) - flow.a**2 * np.exp(1j * alpha) / zp**2 + 1j * g / (2 * np.pi * zp)
    q = np.abs(d / (1 - flow.b**2 / zeta**2))
    return 1 - q**2


def noise_seed(split: str, draw: int, n: int, level: float) -> int:
    return SPLIT_BASE[split] + 1000 * draw + 10 * n + round(1000 * level)


def readings(
    flow: JoukowskiFlow, n: int, alpha: float, u: float, level: float, split: str, draw: int
) -> np.ndarray:
    """Synthetic dp at the ``n`` taps for one noise draw."""
    q = 0.5 * u**2
    clean = q * exact_cp(flow, n, alpha)
    rng = np.random.default_rng(noise_seed(split, draw, n, level))
    return clean + level * q * rng.standard_normal(n)
