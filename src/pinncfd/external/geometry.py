"""NACA 4-digit geometry and panel-method references from AERO-01, for the airfoil family."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from aero.geometry import NACA4, Joukowski
from aero.panel_method_2d import PanelSolution, solve_panels
from matplotlib.path import Path as MplPath

__all__ = [
    "NACA4",
    "Joukowski",
    "naca",
    "contour",
    "inside",
    "surface_samples",
    "panel_solution",
    "solve_panels",
]


def naca(m: float, p: float, t: float) -> NACA4:
    """Closed-trailing-edge NACA 4-digit section with camber m, its position p, thickness t."""
    return NACA4(float(m), float(p), float(t), closed_te=True)


@lru_cache(maxsize=4096)
def _contour(m: float, p: float, t: float, n: int) -> np.ndarray:
    return naca(m, p, t).coordinates(n)


def contour(m, p, t, n: int = 400) -> np.ndarray:
    """Clockwise closed contour (n + 1, 2) from the trailing edge, cosine-spaced."""
    return _contour(float(m), float(p), float(t), int(n))


def inside(m, p, t, pts: np.ndarray) -> np.ndarray:
    """True for points inside (or on) the section."""
    return MplPath(contour(m, p, t)).contains_points(pts, radius=1e-9)


def surface_samples(m, p, t, n: int, rng: np.random.Generator | None = None, n_contour=400):
    """Points on the surface and their outward unit normals.

    Random points are drawn per panel of a cosine-spaced contour (so they cluster at the leading
    and trailing edges as the panels do) at a uniform position along the panel; without ``rng``
    the panel midpoints of an ``n``-panel contour are returned.
    """
    if rng is None:
        nodes = contour(m, p, t, n)
        a, b = nodes[:-1], nodes[1:]
        s = np.full(len(a), 0.5)
    else:
        nodes = contour(m, p, t, n_contour)
        idx = rng.integers(0, len(nodes) - 1, n)
        a, b = nodes[idx], nodes[idx + 1]
        s = rng.random(n)
    pts = a + s[:, None] * (b - a)
    d = b - a
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    # contours run clockwise, so the tangent rotated by +90 degrees points out of the body
    normals = np.column_stack([-d[:, 1], d[:, 0]])
    return pts, normals


def panel_solution(m, p, t, alpha: float, n: int = 200) -> PanelSolution:
    """AERO-01 Hess-Smith solution (chord 1, V = 1) at ``alpha`` [rad]."""
    return solve_panels(contour(m, p, t, n), alpha)
