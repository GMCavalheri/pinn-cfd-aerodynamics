"""Phase 4: the surrogate helpers (correctness, not speed)."""

import math

import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from pinncfd.external.geometry import panel_solution
from pinncfd.surrogate import (
    break_even,
    cl_circulation,
    cl_pressure,
    solve_alpha,
    surface_panels,
    timed,
)

DEG = np.pi / 180


class PanelSurfaceModel(torch.nn.Module):
    """A stand-in 'network' that returns the AERO-01 panel solution's own surface velocity
    (vt along each panel) at its control points, so the pressure integral can be checked
    against the panel method's lift to round-off."""

    def __init__(self, sol):
        super().__init__()
        d = np.diff(sol.nodes, axis=0)
        t = d / np.hypot(d[:, 0], d[:, 1])[:, None]
        self.uv = torch.as_tensor(sol.vt[:, None] * t, dtype=torch.float64)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def velocity(self, xy, q):
        return self.uv[:, :1] + 0 * xy[:, :1], self.uv[:, 1:] + 0 * xy[:, :1]


class VortexModel(torch.nn.Module):
    """Uniform flow plus a clockwise point vortex Gamma at (0.5, 0): circulation known."""

    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def velocity(self, xy, q):
        dx, dy = xy[:, :1] - 0.5, xy[:, 1:2]
        r2 = dx**2 + dy**2
        k = self.gamma / (2 * math.pi)
        return 1.0 + k * dy / r2, -k * dx / r2


class IncidenceModel(torch.nn.Module):
    """Freestream at incidence alpha (an input) plus a clockwise vortex Gamma = pi sin(alpha) at
    (0.5, 0): a smooth stand-in whose surface pressure depends on alpha through q, so the
    derivative Newton needs must flow through the alpha input."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def velocity(self, xy, q):
        a = q[:, 3:4]
        dx, dy = xy[:, :1] - 0.5, xy[:, 1:2]
        r2 = dx**2 + dy**2
        k = torch.sin(a) / 2
        return torch.cos(a) + k * dy / r2, torch.sin(a) - k * dx / r2


def test_surface_panels_match_the_contour():
    xy, nrm, ds = surface_panels(np.array([[0.02, 0.4, 0.12, 0.0]]), 60)
    assert xy.shape == (60, 2) and np.allclose(np.linalg.norm(nrm, axis=1), 1.0)
    assert ds.sum() == pytest.approx(2.03, abs=0.02)  # perimeter of a 12 % section ~ 2.03 c


def test_cl_circulation_recovers_a_known_vortex():
    # Kutta-Joukowski: cl = 2 Gamma for chord 1 and V = 1
    cl = cl_circulation(VortexModel(0.37), np.array([[0.02, 0.4, 0.12, 0.0]] * 3), "cpu")
    assert np.allclose(cl, 0.74, atol=1e-6)


def test_cl_pressure_reproduces_the_panel_methods_pressure_lift():
    sol = panel_solution(0.02, 0.4, 0.12, 4 * DEG, 200)
    params = np.array([[0.02, 0.4, 0.12, 4 * DEG]])
    cl = cl_pressure(PanelSurfaceModel(sol), params, "cpu", n=200)
    # same panels, same integral as AERO-01's cl_pressure, which normalises by the contour's
    # x-extent instead of the chord (1)
    assert cl[0] == pytest.approx(sol.cl_pressure * sol.chord, rel=1e-6)


def test_solve_alpha_matches_a_bracketing_root_finder():
    # Newton with the autograd derivative must find the same root of cl_pressure(alpha) = 0.8 as
    # Brent's method on the same function (symmetric sections, where this stand-in's pressure
    # lift is monotonic in alpha), to Newton's tolerance 1e-5 on cl (float32 round-off in the
    # 120-panel sum is ~1e-6); quadratic convergence from 0 deg takes a handful of steps, which
    # a wrong derivative would not
    model = IncidenceModel()
    params = np.array([[0.0, 0.4, 0.09, 0.0], [0.0, 0.4, 0.15, 0.0]])
    alpha, cl, iters = solve_alpha(model, params, 0.8, "cpu")
    assert np.allclose(cl, 0.8, atol=1e-5) and iters <= 6
    for row, a in zip(params, alpha, strict=True):

        def f(x, row=row):
            return cl_pressure(model, np.array([[*row[:3], x]]), "cpu")[0] - 0.8

        assert a == pytest.approx(brentq(f, 0.0, 0.4, xtol=1e-9), abs=1e-4)


def test_timing_and_break_even():
    out, t = timed(lambda n: sum(range(n)), 1000, repeats=3)
    assert out == 499500 and t >= 0
    assert break_even(600.0, 8e-3, 1e-5) == pytest.approx(600 / (8e-3 - 1e-5))
    assert math.isinf(break_even(600.0, 1e-5, 1e-3))
