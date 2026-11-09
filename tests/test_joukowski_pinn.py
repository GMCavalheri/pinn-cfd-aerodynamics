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
    exact_stream_function,
)
from pinncfd.seed import seed_everything
from pinncfd.training import TrainConfig, train

FLOW = JoukowskiFlow()
ALPHA = np.radians(4.0)
SMALL = dict(n_interior=256, n_body=64, n_far=64, n_eval=256, n_eval_body=100)


class ExactModel(JoukowskiPINN):
    """psi' = exact psi - known part - c, with Gamma and psi_b set from the exact flow and the
    constant c chosen so psi' has zero mean on the far circle (only psi' differences and
    psi - psi_b enter the losses, so c is the one free constant)."""

    shift = 0.0

    def perturbation(self, xy):
        psi = exact_stream_function(xy, self.section, self.alpha)
        return psi - self.known_part(xy) - self.shift


def exact_setup(alpha=ALPHA):
    m = ExactModel(alpha, "plain").double()
    with torch.no_grad():
        m.gamma.fill_(FLOW.circulation(alpha))
    p = P.JoukowskiProblem(m, dtype=torch.float64, **SMALL)
    m.shift = float(m.perturbation(p.xy_far).mean().detach())
    with torch.no_grad():
        m.psi_b.fill_(FLOW.body_value(alpha) - m.shift)
    return m, p


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
    assert losses["body"] < 1e-9  # the body is the streamline psi_b
    # surface samples lie on a 400-panel polygon, not the curve: their normals are off by up to
    # half a panel angle, which leaves a small normal velocity
    assert losses["tangency"] < 1e-4
    # the extrapolated jump vanishes to O(s1 s2): measured 3e-7 at 4 degrees
    assert losses["kutta"] < 1e-5
    # the true disturbance at r = 8 chords is a doublet of order 1e-2 (squared ~1e-5)
    assert losses["far"] < 1e-4


def test_the_metrics_of_the_exact_flow_are_zero():
    m, p = exact_setup()
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
    # ... the jump extrapolated to the edge does not, by five orders of magnitude
    assert ext[1.0] < 1e-6 and ext[0.9] > 1e4 * ext[1.0]


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
    assert p.n_collocation == 256 + 64 + 64 + 4


def test_a_short_run_trains():
    # 300 Adam steps on the CPU: the error must fall well below that of the random
    # initialisation (the first evaluation). This checks the training path, not accuracy.
    seed_everything(0)
    m = JoukowskiPINN(ALPHA, "exact_map", mlp=dict(width=32, depth=3, seed=0))
    p = P.JoukowskiProblem(m, **SMALL)
    cfg = TrainConfig(adam_steps=300, lbfgs_iters=0, eval_every=100, resample_every=100, lr=2e-3)
    res = train(m, p, cfg)
    assert res.final["error"] < 0.6 * res.history[0]["error"]
