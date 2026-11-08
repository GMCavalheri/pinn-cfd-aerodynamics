"""Panel-method baseline from AERO-01 (Hess-Smith source/vortex panels)."""

from __future__ import annotations

import numpy as np
from aero.panel_method_2d import PanelSolution, solve_panels


def cylinder_nodes(n: int, radius: float = 1.0) -> np.ndarray:
    """Regular n-gon inscribed in the circle, clockwise from theta = 0 (AERO-01's ordering)."""
    th = -np.linspace(0, 2 * np.pi, n + 1)
    return np.column_stack([radius * np.cos(th), radius * np.sin(th)])


def solve_cylinder(n: int = 200, gamma: float = 0.0) -> PanelSolution:
    """Non-lifting source-panel solution for the unit cylinder at zero incidence.

    A cylinder has no sharp trailing edge, so the Kutta condition does not fix its circulation.
    For ``gamma != 0`` the circulation is imposed by hand: the vortex strength per unit length
    is set to -gamma / perimeter (AERO-01's sign: positive gamma_panel is counter-clockwise),
    and the sources are solved for with that vortex sheet in place.
    """
    nodes = cylinder_nodes(n)
    if gamma == 0.0:
        return solve_panels(nodes, 0.0, lifting=False)
    from aero.panel_method_2d.hess_smith import _influence_matrices

    a_n, a_t, b_n, b_t, (tx, ty, nx, ny) = _influence_matrices(nodes)
    lengths = np.hypot(*np.diff(nodes, axis=0).T)
    g = -gamma / lengths.sum()
    sigma = np.linalg.solve(a_n, -(nx + g * b_n))
    vt = a_t @ sigma + g * b_t + tx
    return PanelSolution(nodes, 0.0, sigma, float(g), vt)
