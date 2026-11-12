"""RES-01 case 2: the Joukowski PINN, its loss terms and its metrics.

The decisive check is that the *exact* flow (``JoukowskiFlow``) zeroes every loss term up to a
stated, explained residual: then a network that trains badly has failed the method, not a
mis-posed problem.
"""

import numpy as np
import pytest
import torch

from pinncfd.pinn_joukowski import JoukowskiFlow
from pinncfd.pinn_joukowski import problem as P
from pinncfd.pinn_joukowski.model import (
    FEATURES,
    JoukowskiPINN,
    circle_plane,
    exact_basis,
    exact_stream_function,
)
from pinncfd.seed import seed_everything
from pinncfd.training import TrainConfig, train

FLOW = JoukowskiFlow()
ALPHA = np.radians(4.0)
SMALL = dict(n_interior=256, n_body=64, n_far=64, n_eval=256, n_eval_body=100)


class ExactModel(JoukowskiPINN):
    """psi' = exact basis flow - known part - c for each of the three basis flows, with c chosen
    so psi' has zero mean on the far circle (only psi' differences and psi - psi_b enter the
    losses, so c is the one free constant of each), and psi_b the exact body values. With
    ``exact_gamma`` the circulation is the exact one instead of the Kutta circulation of the
    basis flows (which carries the small bias of the extrapolated Kutta condition)."""

    shift = 0.0
    exact_gamma = False

    def perturbation(self, xy):
        return exact_basis(xy, self.section) - self.known_part(xy) - self.shift

    def gamma_for(self, a):
        if self.exact_gamma:
            a = float(torch.as_tensor(a).detach())
            return torch.as_tensor(FLOW.circulation(a), dtype=torch.float64)
        return super().gamma_for(a)


def exact_setup(alpha=ALPHA, exact_gamma=False):
    m = ExactModel(alpha, "plain").double()
    m.exact_gamma = exact_gamma
    p = P.JoukowskiProblem(m, dtype=torch.float64, **SMALL)
    m.shift = m.perturbation(p.xy_far).mean(dim=0).detach()
    with torch.no_grad():
        m.psi_b.copy_(m.basis(p.xy_body).mean(dim=0))
    m.update_kutta()
    return m, p


def test_the_exact_basis_flows_superpose_to_the_exact_flow():
    xy = torch.tensor(P.exterior_points(FLOW.section, 200, np.random.default_rng(1)))
    b = exact_basis(xy, FLOW.section)
    psi = np.cos(ALPHA) * b[:, :1] + np.sin(ALPHA) * b[:, 1:2] + FLOW.circulation(ALPHA) * b[:, 2:]
    diff = (psi - exact_stream_function(xy, FLOW.section, ALPHA)).numpy()
    assert np.ptp(diff) < 1e-12  # equal up to a constant


def test_torch_circle_plane_matches_the_numpy_map():
    rng = np.random.default_rng(0)
    xy = P.exterior_points(FLOW.section, 300, rng)
    w = circle_plane(torch.tensor(xy, dtype=torch.float64), FLOW.section).numpy().ravel()
    assert np.allclose(w, FLOW.to_circle(xy[:, 0], xy[:, 1]) - FLOW.zeta0, atol=1e-12)


@pytest.mark.parametrize("alpha", np.radians([0.0, 4.0, 8.0]))
def test_the_exact_flow_zeroes_every_loss_term(alpha):
    m, p = exact_setup(alpha)
    losses = {k: float(v.detach()) for k, v in p.losses(m).items()}
    assert losses["pde"] < 1e-20  # exactly harmonic (float64 round-off)
    assert losses["body"] < 1e-9  # the body is a streamline of every basis flow
    # surface samples lie on a 400-panel polygon, not the curve: their normals are off by up to
    # half a panel angle, which leaves a small normal velocity. The vertical-freestream basis
    # flow runs fast round the nose and the cusp, so its share dominates: measured 1.0e-3 of the
    # total 1.2e-3 (largest normal velocity 0.15, at the cusp)
    assert losses["tangency"] < 3e-3
    # the true disturbance at r = 8 chords is a doublet of order 1e-2 (squared ~1e-5), in each
    # of three basis flows
    assert losses["far"] < 1e-3


@pytest.mark.parametrize("alpha", np.radians([0.0, 4.0, 8.0]))
def test_the_kutta_circulation_of_the_exact_basis_flows_is_the_exact_circulation(alpha):
    # a straight line in s = sqrt(1 - x) over x = 0.85..0.985 leaves the curvature of the
    # physical flow's jump at the edge: up to 4.5e-3 at 8 degrees, which over the jump's
    # sensitivity to Gamma (about 2.7) is under 2e-3 in Gamma, i.e. under 0.3 %
    m, _ = exact_setup(alpha)
    assert float(m.gamma) == pytest.approx(FLOW.circulation(alpha), rel=3e-3, abs=2e-3)


def test_the_metrics_of_the_exact_flow_are_zero():
    m, p = exact_setup(exact_gamma=True)
    for split in ("val", "test"):
        out = p.evaluate(m, split)
        assert out["error"] < 1e-12 and out["cp_max_abs"] < 1e-10
        assert out["cl_err_circulation"] < 1e-10 and out["cl_err_pressure"] < 1e-10


def _jump(stations_vt):
    return stations_vt[0] - stations_vt[1]


def test_the_extrapolated_kutta_condition_is_unbiased_and_the_plain_one_is_not():
    # Potential flows around the section with circulation k Gamma_exact (all exact solutions of
    # Laplace with the body a streamline; only k = 1 satisfies the Kutta condition).
    sec, a = FLOW.section, ALPHA
    xk = P.kutta_points(sec)

    def vt(k):
        zeta = FLOW.to_circle(xk[:, 0], xk[:, 1])
        zp = zeta - FLOW.zeta0
        g = k * FLOW.foil.circulation(a)
        d = np.exp(-1j * a) - FLOW.a**2 * np.exp(1j * a) / zp**2 + 1j * g / (2 * np.pi * zp)
        w = d / (1 - FLOW.b**2 / zeta**2)
        return (w.real * xk[:, 2] - w.imag * xk[:, 3]).reshape(2, -1)

    raw = {k: np.mean(_jump(vt(k)) ** 2) for k in (1.0, 0.9)}
    ext = {k: P.trailing_edge_jump(*vt(k)) ** 2 for k in (1.0, 0.9)}
    # the station jump itself rewards 10 % too little circulation ...
    assert raw[0.9] < raw[1.0]
    # ... the jump extrapolated to the edge does not, by four orders of magnitude
    assert ext[1.0] < 1e-6 and ext[0.9] > 1e4 * ext[1.0]


def test_the_exact_perturbations_have_no_net_flux_through_the_rings():
    # psi' of each exact basis flow is harmonic and bounded outside the section, so its flux
    # through every enclosing curve is zero (Gauss); the ring means are a 128-point trapezoidal
    # rule of a smooth periodic function, accurate to round-off
    m, p = exact_setup()

    def ring_flux(psi):
        xr = p.xy_rings
        out = []
        for k in range(3):
            g = torch.autograd.grad(psi(xr)[:, k].sum(), xr)[0]
            r_dr = (xr[:, :1] - 0.5) * g[:, :1] + xr[:, 1:2] * g[:, 1:2]
            out.append(r_dr.view(len(P.FLUX_RADII), -1).mean(dim=1).detach().numpy())
        return np.array(out)

    assert np.abs(ring_flux(m.perturbation)).max() < 1e-10
    # a psi' that carried the circulation itself (the whole circulation basis flow) would show
    # a flux of 1 / (2 pi) on every ring: exactly what the flux term forbids
    flux = ring_flux(lambda xy: exact_basis(xy, FLOW.section))
    assert np.allclose(flux[2], 1 / (2 * np.pi), rtol=1e-8) and np.abs(flux[:2]).max() < 1e-10


def test_validation_and_test_sets_are_disjoint():
    _, p = exact_setup()
    v, t = p.refs["val"], p.refs["test"]
    for key in ("field_xy", "body_xy"):
        gap = np.linalg.norm(v[key][:, None, :] - t[key][None, :, :], axis=2)
        assert gap.min() > 1e-9  # no point is shared
    # every evaluation point is in the fluid, off the surface
    for ref in (v, t):
        assert not FLOW.section.inside(ref["field_xy"]).any()
        assert P.distance_to_surface(FLOW.section, ref["field_xy"]).min() >= P.EVAL_MIN_DISTANCE


@pytest.mark.parametrize("features", FEATURES)
def test_every_feature_type_builds_and_gives_all_loss_terms(features):
    m = JoukowskiPINN(ALPHA, features, mlp=dict(width=16, depth=2))
    p = P.JoukowskiProblem(m, **SMALL)
    losses = p.losses(m)
    assert set(losses) == set(P.DEFAULT_WEIGHTS)
    assert all(torch.isfinite(v) for v in losses.values())
    # interior, body, far, Kutta (8 stations x 2 sides), flux rings (3 x 128)
    assert p.n_collocation == 256 + 64 + 64 + 16 + 3 * 128


def test_a_short_run_trains():
    # 300 Adam steps on the CPU: the error must fall well below that of the random
    # initialisation (the first evaluation). This checks the training path, not accuracy.
    seed_everything(0)
    m = JoukowskiPINN(ALPHA, "exact_map", mlp=dict(width=32, depth=3, seed=0))
    p = P.JoukowskiProblem(m, **SMALL)
    cfg = TrainConfig(adam_steps=300, lbfgs_iters=0, eval_every=100, resample_every=100, lr=2e-3)
    res = train(m, p, cfg)
    assert res.final["error"] < 0.6 * res.history[0]["error"]
