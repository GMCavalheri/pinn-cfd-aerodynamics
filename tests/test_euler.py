"""Phase 5: exact Euler solutions, the residual operator and the problem plumbing."""

import numpy as np
import pytest
import torch

from pinncfd.pinn_euler_2d import EulerPINN, RinglebProblem, WedgeProblem, residual
from pinncfd.pinn_euler_2d.exact import (
    GAMMA,
    K_MAX,
    K_MIN,
    Q0,
    oblique_shock,
    ringleb,
    ringleb_boundary,
    ringleb_samples,
    theta_beta_m,
    wedge_field,
)


def _flux_np(rho, u, v, p):
    E = p / (GAMMA - 1) + 0.5 * rho * (u**2 + v**2)
    F = np.array([rho * u, rho * u**2 + p, rho * u * v, u * (E + p)])
    G = np.array([rho * v, rho * u * v, rho * v**2 + p, v * (E + p)])
    return F, G


@pytest.mark.parametrize("q,k,b", [(0.6, 0.8, 1), (0.95, 1.1, -1), (1.05, 1.15, 1), (0.7, 1.0, -1)])
def test_ringleb_satisfies_the_steady_euler_equations(q, k, b):
    # divergence of the fluxes through the hodograph Jacobian, by central differences
    # (truncation ~ h^2 ~ 1e-10, round-off ~ eps/h ~ 1e-11); includes a supersonic point
    h = 1e-5

    def at(qq, kk):
        x, y, *w = ringleb(qq, kk, b)
        F, G = _flux_np(*w)
        return np.array([x, y]), F, G

    (Xqp, Fqp, Gqp), (Xqm, Fqm, Gqm) = at(q + h, k), at(q - h, k)
    (Xkp, Fkp, Gkp), (Xkm, Fkm, Gkm) = at(q, k + h), at(q, k - h)
    Xq, Xk = (Xqp - Xqm) / (2 * h), (Xkp - Xkm) / (2 * h)
    inv = np.linalg.inv(np.column_stack([Xq, Xk]))  # rows: dq/d(x,y), dk/d(x,y)
    Fq, Fk = (Fqp - Fqm) / (2 * h), (Fkp - Fkm) / (2 * h)
    Gq, Gk = (Gqp - Gqm) / (2 * h), (Gkp - Gkm) / (2 * h)
    div = Fq * inv[0, 0] + Fk * inv[1, 0] + Gq * inv[0, 1] + Gk * inv[1, 1]
    assert np.abs(div).max() < 1e-6


def test_ringleb_streamlines_are_lines_of_constant_k():
    q, k, h = np.array([0.6, 0.9, 1.1]), np.array([0.8, 1.0, 1.15]), 1e-7
    for b in (1, -1):
        x0, y0, _, u, v, _ = ringleb(q, k, b)
        x1, y1, *_ = ringleb(q + h, k, b)
        cross = (x1 - x0) * v - (y1 - y0) * u  # tangent of k = const parallel to the velocity
        assert np.allclose(cross / h, 0.0, atol=1e-5)
    # sonic speed sqrt(2/(gamma + 1)) separates sub- and supersonic points
    _, _, rho, u, v, p = ringleb(1.0, 1.1, 1)
    assert np.hypot(u, v) / np.sqrt(GAMMA * p / rho) > 1


def test_ringleb_sampling_stays_in_the_domain():
    rng = np.random.default_rng(0)
    q, k, b = ringleb_samples(500, rng).T
    assert len(q) == 500 and np.all((k >= K_MIN) & (k <= K_MAX) & (q >= Q0) & (q <= k))
    q, k, _ = ringleb_boundary(300, rng).T
    assert np.all(np.isclose(k, K_MIN) | np.isclose(k, K_MAX) | np.isclose(q, Q0))


def test_oblique_shock_matches_tabulated_values():
    # M = 2, theta = 10 deg: beta = 39.3 deg, p2/p1 = 1.71, rho2/rho1 = 1.46, M2 = 1.64
    # (NACA Report 1135, oblique-shock charts; Anderson, Modern Compressible Flow, Ch. 4);
    # tolerances match the tabulated rounding
    s = oblique_shock(np.radians(10.0), 2.0)
    assert np.degrees(s["beta"]) == pytest.approx(39.3, abs=0.05)
    assert s["p_ratio"] == pytest.approx(1.71, abs=0.005)
    assert s["rho_ratio"] == pytest.approx(1.46, abs=0.005)
    assert s["mach2"] == pytest.approx(1.64, abs=0.005)
    assert theta_beta_m(s["beta"], 2.0) == pytest.approx(np.radians(10.0))


def test_oblique_shock_conserves_normal_fluxes():
    # Rankine-Hugoniot: mass, momentum and energy fluxes through the shock are continuous
    theta = np.radians(10.0)
    s = oblique_shock(theta, 2.0)
    n = np.array([np.sin(s["beta"]), -np.cos(s["beta"])])  # shock normal
    fluxes = []
    for w in (s["state1"], s["state2"]):
        F, G = _flux_np(*w)
        fluxes.append(F * n[0] + G * n[1])
    assert np.allclose(fluxes[0], fluxes[1], rtol=1e-12)
    # and the post-shock flow is parallel to the wall
    assert np.arctan2(s["state2"][2], s["state2"][1]) == pytest.approx(theta)


def test_residual_of_known_fields():
    xy = torch.rand(50, 2, dtype=torch.float64, requires_grad=True)

    class Field(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, xy):
            return self.f(xy)

    uniform = Field(
        lambda xy: torch.stack(
            [1 + 0 * xy[:, 0], 2 + 0 * xy[:, 0], 0.5 + 0 * xy[:, 0], 0.7 + 0 * xy[:, 0]], 1
        )
    )
    assert torch.allclose(residual(uniform, xy), torch.zeros(50, 4, dtype=torch.float64))
    # rho = 1 + x, u = 1, v = 0, p = 1: mass residual d(rho u)/dx = 1, x-momentum also 1
    ramp = Field(
        lambda xy: torch.stack([1 + xy[:, 0], 1 + 0 * xy[:, 0], 0 * xy[:, 0], 1 + 0 * xy[:, 0]], 1)
    )
    r = residual(ramp, xy)
    assert torch.allclose(r[:, 0], torch.ones(50, dtype=torch.float64))
    assert torch.allclose(r[:, 1], torch.ones(50, dtype=torch.float64))
    # artificial viscosity of a field with Laplacian 2 in rho subtracts 2 nu
    quad = Field(
        lambda xy: torch.stack([1 + xy[:, 0] ** 2, 0 * xy[:, 0], 0 * xy[:, 0], 1 + 0 * xy[:, 0]], 1)
    )
    assert torch.allclose(
        residual(quad, xy, nu=0.1)[:, 0], torch.full((50,), -0.2, dtype=torch.float64)
    )


def test_model_keeps_density_and_pressure_positive():
    m = EulerPINN(mlp=dict(width=8, depth=2), ref=[1.0, 2.0, 1.0, 0.7])
    w = m(torch.randn(100, 2) * 50)
    assert (w[:, 0] > 0).all() and (w[:, 3] > 0).all()
    w0 = EulerPINN(mlp=dict(width=8, depth=2), ref=[1.0, 2.0, 1.0, 0.7])
    with torch.no_grad():
        for p in w0.parameters():
            p.zero_()
    # with a zero network the output is the reference density and pressure
    assert torch.allclose(w0(torch.zeros(1, 2)), torch.tensor([[1.0, 0.0, 0.0, 0.7]]), atol=1e-4)


def test_shock_angle_detector_on_the_exact_field():
    class Exact(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d = torch.nn.Parameter(torch.zeros(1))

        def forward(self, xy):
            w = wedge_field(*xy.detach().cpu().numpy().T, np.radians(10), 2.0)
            return torch.as_tensor(w, dtype=torch.float32)

    prob = WedgeProblem(Exact(), n_interior=64, n_boundary=32, device="cpu")
    m = prob.evaluate(Exact())
    assert m["error"] == pytest.approx(0.0, abs=1e-6)
    assert m["beta_err_deg"] < 0.05  # grid of 2001 points in y


def test_problems_produce_finite_losses():
    for cls in (RinglebProblem, WedgeProblem):
        m = EulerPINN(mlp=dict(width=8, depth=2))
        prob = cls(m, n_interior=64, n_boundary=32, device="cpu")
        assert all(torch.isfinite(v) for v in prob.losses(m).values())
        assert np.isfinite(prob.evaluate(m)["error"])
