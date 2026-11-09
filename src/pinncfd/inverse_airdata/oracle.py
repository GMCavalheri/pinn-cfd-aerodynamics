"""The model-based estimator: fit (alpha, U) to the tap readings with the exact pressure model.

This is what a calibrated flush air data system does when its pressure model is right, and it
is the noise floor a PINN estimate is judged against: the PINN has to learn the flow *and* the
air data from the same few readings, the oracle only the air data.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from pinncfd.inverse_airdata.taps import exact_cp
from pinncfd.pinn_joukowski.exact import JoukowskiFlow


def fit(flow: JoukowskiFlow, n: int, dp: np.ndarray, alpha0: float = 0.0, u0: float = 1.0):
    """Least-squares (alpha, U) from tap readings ``dp``; returns (alpha, U)."""

    def residual(p):
        alpha, u = p
        return 0.5 * u**2 * exact_cp(flow, n, alpha) - dp

    sol = least_squares(residual, [alpha0, u0], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return float(sol.x[0]), float(abs(sol.x[1]))
