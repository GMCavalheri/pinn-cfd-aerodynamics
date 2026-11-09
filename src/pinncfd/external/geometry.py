"""Section geometry and panel-method references from AERO-01.

NACA 4-digit helpers serve the airfoil family (Phase 2 on); :class:`JoukowskiSection` serves
the RES-01 Joukowski case, whose exact flow lives in ``pinncfd.pinn_joukowski.exact``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from aero.geometry import NACA4, Joukowski
from aero.panel_method_2d import PanelSolution, solve_panels
from matplotlib.path import Path as MplPath

__all__ = [
    "NACA4",
    "Joukowski",
    "JoukowskiSection",
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


# ---- Joukowski sections (RES-01 case 2) ------------------------------------------------------


@lru_cache(maxsize=16)
def _joukowski(mx: float, my: float):
    foil = Joukowski(mx, my)
    return foil, foil.chord, foil.x_le  # chord and x_le scan 20001 points: compute once


@lru_cache(maxsize=64)
def _joukowski_contour(mx: float, my: float, n: int) -> np.ndarray:
    return _joukowski(mx, my)[0].coordinates(n)


def _samples_on(nodes: np.ndarray, n: int, rng: np.random.Generator | None):
    """Points on a clockwise contour and their outward normals (as ``surface_samples``)."""
    if rng is None:
        a, b = nodes[:-1], nodes[1:]
        s = np.full(len(a), 0.5)
    else:
        idx = rng.integers(0, len(nodes) - 1, n)
        a, b = nodes[idx], nodes[idx + 1]
        s = rng.random(n)
    pts = a + s[:, None] * (b - a)
    d = b - a
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return pts, np.column_stack([-d[:, 1], d[:, 0]])


@dataclass(frozen=True)
class JoukowskiSection:
    """A Joukowski airfoil normalised to unit chord with its leading edge at x = 0.

    The default is AERO-01's validation section, ``Joukowski(0.1, 0.1)`` (thickness and camber
    offsets of the generating circle), with its cusped trailing edge at (1, 0). Contours run
    clockwise from the trailing edge, lower surface first, with nodes equispaced in the circle
    angle (clustered at both edges).
    """

    mx: float = 0.1
    my: float = 0.1

    @property
    def foil(self) -> Joukowski:
        return _joukowski(self.mx, self.my)[0]

    @property
    def chord(self) -> float:
        """Chord of the unnormalised section (the normalised one has chord 1)."""
        return _joukowski(self.mx, self.my)[1]

    @property
    def x_le(self) -> float:
        """Leading-edge x of the unnormalised section."""
        return _joukowski(self.mx, self.my)[2]

    def contour(self, n: int = 400) -> np.ndarray:
        return _joukowski_contour(float(self.mx), float(self.my), int(n))

    def inside(self, pts: np.ndarray) -> np.ndarray:
        return MplPath(self.contour(400)).contains_points(pts, radius=1e-9)

    def surface_samples(self, n: int, rng: np.random.Generator | None = None, n_contour=400):
        """Random points per panel of the contour (or the ``n`` panel midpoints without ``rng``)
        and their outward unit normals."""
        nodes = self.contour(n if rng is None else n_contour)
        return _samples_on(nodes, n, rng)

    def surfaces(self, xs):
        """(xu, yu, xl, yl): upper and lower surface points at the stations ``xs``."""
        xs = np.asarray(xs, dtype=float)
        c = self.contour(4000)
        i_le = int(np.argmin(c[:, 0]))
        lower, upper = c[: i_le + 1][::-1], c[i_le:]  # both from the leading edge back
        yl = np.interp(xs, lower[:, 0], lower[:, 1])
        yu = np.interp(xs, upper[:, 0], upper[:, 1])
        return xs, yu, xs.copy(), yl

    def center(self) -> tuple[float, float]:
        """A point well inside the section: the camber line at mid-chord."""
        _, yu, _, yl = self.surfaces([0.5])
        return 0.5, float((yu[0] + yl[0]) / 2)
