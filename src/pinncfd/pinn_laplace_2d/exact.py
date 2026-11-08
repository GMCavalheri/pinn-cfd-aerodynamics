"""Closed-form potential flow past a circular cylinder (radius 1, freestream 1 along +x).

Uniform flow + doublet gives the non-lifting cylinder; adding a point vortex of clockwise
circulation Gamma gives the lifting cylinder (Anderson, *Fundamentals of Aerodynamics*,
Secs. 3.12-3.15). With r^2 = x^2 + y^2:

    psi = y (1 - 1/r^2) + (Gamma / 2 pi) ln r          (stream function)
    phi = x (1 + 1/r^2)                                (velocity potential, Gamma = 0 only)
    u = d psi/dy,  v = -d psi/dx

On the surface V_theta = -2 sin(theta) - Gamma/(2 pi), so Cp = 1 - (2 sin(theta) + Gamma/(2 pi))^2,
and the Kutta-Joukowski theorem gives lift L' = rho V Gamma, i.e. cl = Gamma referred to the
diameter (chord 2).
"""

from __future__ import annotations

import numpy as np
import torch


def _log(a):
    # the formulas work on NumPy arrays and on torch tensors (for autograd checks)
    return torch.log(a) if isinstance(a, torch.Tensor) else np.log(a)


def stream_function(x, y, gamma: float = 0.0):
    r2 = x**2 + y**2
    return y * (1.0 - 1.0 / r2) + gamma / (4 * np.pi) * _log(r2)


def potential(x, y):
    return x * (1.0 + 1.0 / (x**2 + y**2))


def velocity(x, y, gamma: float = 0.0):
    r2 = x**2 + y**2
    u = 1.0 - 1.0 / r2 + 2 * y**2 / r2**2 + gamma / (2 * np.pi) * y / r2
    v = -2 * x * y / r2**2 - gamma / (2 * np.pi) * x / r2
    return u, v


def surface_cp(theta, gamma: float = 0.0):
    return 1.0 - (2 * np.sin(theta) + gamma / (2 * np.pi)) ** 2


def cl(gamma: float = 0.0) -> float:
    """Lift coefficient referred to the diameter."""
    return float(gamma)


def cl_from_cp(theta, cp) -> float:
    """cl = -(1/2) integral of Cp sin(theta) d(theta) for equispaced theta over [0, 2 pi).

    The periodic trapezoidal rule is spectrally accurate here.
    """
    return float(-0.5 * np.mean(cp * np.sin(theta)) * 2 * np.pi)
