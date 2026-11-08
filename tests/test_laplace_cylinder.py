"""Phase 1: potential flow past a cylinder, exact solution and PINN."""

import numpy as np
import pytest
import torch

from pinncfd.external.panel import solve_cylinder
from pinncfd.pinn_laplace_2d import CylinderPINN, CylinderProblem, exact
from pinncfd.seed import seed_everything
from pinncfd.training import TrainConfig, train

GAMMA = 2 * np.pi  # stagnation points at sin(theta) = -1/2


@pytest.mark.parametrize("gamma", [0.0, GAMMA])
def test_exact_stream_function_is_harmonic_and_matches_the_velocity(gamma):
    # Anderson Secs. 3.13-3.15: psi = y (1 - 1/r^2) + (Gamma / 2 pi) ln r, u = psi_y, v = -psi_x
    g = torch.Generator().manual_seed(0)
    r = 1 + 4 * torch.rand(200, 1, generator=g, dtype=torch.float64)
    th = 2 * np.pi * torch.rand(200, 1, generator=g, dtype=torch.float64)
    xy = torch.cat([r * torch.cos(th), r * torch.sin(th)], 1).requires_grad_()
    psi = exact.stream_function(xy[:, :1], xy[:, 1:], gamma)
    from pinncfd.autodiff import grad, laplacian

    assert torch.allclose(laplacian(psi, xy), torch.zeros_like(psi), atol=1e-12)
    gp = grad(psi, xy)
    u, v = exact.velocity(xy[:, :1], xy[:, 1:], gamma)
    assert torch.allclose(gp[:, 1:], u) and torch.allclose(-gp[:, :1], v)


@pytest.mark.parametrize("gamma", [0.0, GAMMA])
def test_exact_surface_flow(gamma):
    th = np.linspace(0, 2 * np.pi, 73, endpoint=False)
    x, y = np.cos(th), np.sin(th)
    # the body is the streamline psi = 0, and the flow is tangent to it
    assert np.allclose(exact.stream_function(x, y, gamma), 0.0, atol=1e-14)
    u, v = exact.velocity(x, y, gamma)
    assert np.allclose(u * x + v * y, 0.0, atol=1e-14)
    assert np.allclose(1 - u**2 - v**2, exact.surface_cp(th, gamma))
    # Kutta-Joukowski: cl = Gamma (diameter reference), recovered by pressure integration
    assert exact.cl_from_cp(th, exact.surface_cp(th, gamma)) == pytest.approx(exact.cl(gamma))


def test_potential_and_stream_function_give_the_same_velocity():
    from pinncfd.autodiff import grad

    xy = torch.tensor([[1.5, 0.3], [-2.0, 1.1], [0.2, -3.0]], dtype=torch.float64)
    xy.requires_grad_()
    gphi = grad(exact.potential(xy[:, :1], xy[:, 1:]), xy)
    u, v = exact.velocity(xy[:, :1], xy[:, 1:])
    assert torch.allclose(gphi[:, :1], u) and torch.allclose(gphi[:, 1:], v)


@pytest.mark.parametrize("gamma", [0.0, GAMMA])
def test_hard_constraint_ansatz_satisfies_the_boundary_conditions(gamma):
    model = CylinderPINN(formulation="psi", hard=True, gamma=gamma, mlp=dict(width=8, depth=2))
    th = torch.linspace(0, 2 * np.pi, 50)[:, None]
    ring = torch.cat([torch.cos(th), torch.sin(th)], 1)
    with torch.no_grad():
        assert torch.allclose(model(ring), torch.zeros(50, 1), atol=1e-6)
        psi_far = model(5.0 * ring).double().numpy()
    ref = exact.stream_function(5.0 * ring[:, :1].double(), 5.0 * ring[:, 1:].double(), gamma)
    assert np.allclose(psi_far, ref.numpy(), atol=1e-5)


def test_phi_formulation_rejects_circulation():
    with pytest.raises(ValueError):
        CylinderPINN(formulation="phi", hard=False, gamma=1.0)


@pytest.mark.parametrize("gamma", [0.0, GAMMA])
def test_short_cpu_training_converges(gamma):
    # 300 Adam steps + 100 L-BFGS iterations of a 3x32 network on 1024 points: a run this short
    # reaches about 1 % velocity error (0.6-1.2 % measured); 5 % leaves margin for BLAS
    # differences between machines while still failing for a broken loss or ansatz.
    seed_everything(0)
    model = CylinderPINN(formulation="psi", hard=True, gamma=gamma, mlp=dict(width=32, depth=3))
    problem = CylinderProblem(
        model, n_interior=1024, n_boundary=128, n_eval_r=24, n_eval_theta=48, n_surface=90
    )
    cfg = TrainConfig(adam_steps=300, lbfgs_iters=100, lr=3e-3, lr_final=1e-3, resample_every=0)
    res = train(model, problem, cfg)
    assert res.history[0]["error"] > 0.1
    assert res.final["error"] < 0.05
    assert res.final["cl"] == pytest.approx(exact.cl(gamma), abs=0.3)


def _panel_cp_error(n, gamma):
    sol = solve_cylinder(n, gamma)
    xc, yc = sol.control_points.T
    return sol, np.abs(sol.cp - exact.surface_cp(np.arctan2(yc, xc), gamma)).max()


def test_panel_baseline_on_the_cylinder():
    # On a regular polygon, constant-source panels reproduce the exact Cp at the control points
    # to round-off (AERO-01, tests/test_panel_method.py).
    _, err = _panel_cp_error(128, 0.0)
    assert err < 1e-10


def test_lifting_panel_baseline_converges_at_first_order():
    # With the circulation imposed as a uniform vortex sheet, cl = Gamma holds by construction
    # (Kutta-Joukowski), but the sheet's tangential velocity at the polygon's control points
    # differs from the point vortex's by O(1/N): measured max |dCp| 0.060, 0.031, 0.016, 0.008
    # for N = 64...512, so the error must halve with each doubling.
    errs = []
    for n in (64, 128, 256):
        sol, err = _panel_cp_error(n, GAMMA)
        assert sol.cl == pytest.approx(GAMMA, abs=1e-12)
        errs.append(err)
    assert errs[0] / errs[1] == pytest.approx(2.0, rel=0.1)
    assert errs[1] / errs[2] == pytest.approx(2.0, rel=0.1)
