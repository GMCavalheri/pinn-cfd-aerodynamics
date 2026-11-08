"""Phase 2: the airfoil family, the model's known part, the sampling and the PINN problem."""

import numpy as np
import pytest
import torch

from pinncfd.autodiff import laplacian
from pinncfd.external.geometry import contour, inside, naca, panel_solution, surface_samples
from pinncfd.pinn_airfoil_flowfield import AirfoilPINN, AirfoilProblem, build_reference, family
from pinncfd.pinn_airfoil_flowfield.model import R_FAR, X0, camber_at_midchord, compress
from pinncfd.pinn_airfoil_flowfield.problem import (
    MIN_DISTANCE,
    cl_from_cp,
    exterior_points,
    field_points,
)
from pinncfd.seed import seed_everything
from pinncfd.training import TrainConfig, train

DEG = np.pi / 180


def test_latin_hypercube_has_one_point_per_stratum():
    pts = family.latin_hypercube(10, np.random.default_rng(0))
    u = (pts - family.LOW) / (family.HIGH - family.LOW)
    for j in range(u.shape[1]):
        assert sorted(np.floor(u[:, j] * 10).astype(int)) == list(range(10))


def test_split_cases_lie_in_the_box_and_do_not_overlap():
    tr, va, te = family.train_cases(), family.val_cases(), family.test_cases()
    assert (len(tr), len(va), len(te)) == (40, 8, 24)
    for c in tr + va + te:
        assert np.all(c.params >= family.LOW - 1e-12) and np.all(c.params <= family.HIGH + 1e-12)
    train_set = {tuple(np.round(c.params, 6)) for c in tr}
    assert not train_set & {tuple(np.round(c.params, 6)) for c in va + te}
    assert te[0].name == "NACA 0012 α=-3.0°"
    # every extrapolation case leaves the box in at least one parameter
    for c in family.extrapolation_cases():
        assert np.any(c.params < family.LOW) or np.any(c.params > family.HIGH)


@pytest.mark.parametrize("code", ["2412", "4415", "5318"])
def test_camber_at_midchord_matches_the_naca_definition(code):
    sec = naca(int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100)
    assert float(camber_at_midchord(sec.m, sec.p)) == pytest.approx(float(sec.camber(0.5)))


def test_surface_samples_lie_on_the_surface_with_outward_normals():
    rng = np.random.default_rng(0)
    m, p, t = 0.04, 0.4, 0.12
    pts, nrm = surface_samples(m, p, t, 500, rng)
    nodes = contour(m, p, t, 400)
    dist = np.min(np.linalg.norm(pts[:, None] - nodes[None], axis=2), axis=1)
    assert dist.max() < 0.01  # within a panel length of a contour node
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)
    away = pts + 1e-3 * nrm
    back = pts - 1e-3 * nrm
    # near the cusp-like trailing edge a step of 1e-3 can cross the thin section; skip x > 0.99
    keep = pts[:, 0] < 0.99
    assert not inside(m, p, t, away[keep]).any()
    assert inside(m, p, t, back[keep]).mean() > 0.99


def test_collocation_and_reference_points_are_outside_the_body():
    rng = np.random.default_rng(1)
    m, p, t = 0.02, 0.4, 0.12
    pts = exterior_points(m, p, t, 2000, rng)
    assert len(pts) == 2000 and not inside(m, p, t, pts).any()
    assert np.hypot(pts[:, 0] - X0, pts[:, 1]).max() <= R_FAR + 1e-9
    fp = field_points(m, p, t, 300, rng)
    nodes = contour(m, p, t, 400)
    assert np.min(np.linalg.norm(fp[:, None] - nodes[None], axis=2)) >= MIN_DISTANCE


def test_known_part_is_harmonic_and_gives_the_freestream():
    model = AirfoilPINN(mlp=dict(width=8, depth=1), head=dict(width=4, depth=1))
    xy = torch.tensor([[2.0, 1.0], [-1.0, -0.5], [0.5, 3.0]], requires_grad=True)
    q = torch.tensor([[0.04, 0.4, 0.12, 5 * DEG]] * 3)
    gamma = torch.full((3, 1), 0.3)
    psi = model.known_part(xy, q, gamma)
    assert torch.allclose(laplacian(psi, xy), torch.zeros(3, 1), atol=1e-5)
    # without circulation the known part is the uniform stream at incidence alpha
    xy0 = xy.detach().clone().requires_grad_()
    psi0 = model.known_part(xy0, q, torch.zeros(3, 1))
    (g,) = torch.autograd.grad(psi0.sum(), xy0)
    assert torch.allclose(g[:, 1], torch.full((3,), np.cos(5 * DEG), dtype=torch.float32))
    assert torch.allclose(-g[:, 0], torch.full((3,), np.sin(5 * DEG), dtype=torch.float32))


def test_compression_keeps_direction_and_bounds_the_far_field():
    xy = torch.tensor([[X0 + 0.1, 0.0], [X0, 2.0], [X0 - R_FAR, 0.0]])
    c = compress(xy)
    assert c[0, 0] == pytest.approx(np.log1p(0.2), rel=1e-5)  # r = 0.1, a = 0.5
    assert c[1, 0] == pytest.approx(0.0, abs=1e-6) and c[1, 1] > 0
    assert torch.linalg.norm(c[2]) == pytest.approx(np.log1p(16.0), rel=1e-5)


def test_cl_from_pressure_matches_the_panel_method():
    sol = panel_solution(0.02, 0.4, 0.12, 4 * DEG, 200)
    cl = cl_from_cp(sol.cp, sol.normals, sol.lengths, sol.alpha)
    # the same integral as AERO-01's PanelSolution.cl_pressure, which divides by the contour's
    # x-extent (slightly over 1 for a cambered section) where this one uses the chord, 1
    assert cl == pytest.approx(sol.cl_pressure * sol.chord, rel=1e-12)
    # and close to the Kutta-Joukowski lift; AERO-01 reports the two within ~1 % at N = 200
    assert cl == pytest.approx(sol.cl, rel=2e-2)


def test_short_training_reduces_the_error():
    # 150 Adam steps on 4 shapes x 96 points with panel data from 6 training cases: the network
    # cannot be accurate yet, but its velocity error on the validation cases must fall well
    # below the untrained network's; a broken loss or sign convention would not.
    seed_everything(0)
    train_cases = family.train_cases()[:6]
    model = AirfoilPINN(mlp=dict(width=32, depth=3), head=dict(width=16, depth=1))
    ref = build_reference(family.val_cases()[:2], n_field=64, n_panels=120)
    problem = AirfoilProblem(
        model,
        train=train_cases,
        evaluation=ref,
        n_cases=4,
        n_interior=96,
        n_body=48,
        n_far=16,
        n_data_cases=4,
        n_data=32,
        device="cpu",
    )
    cfg = TrainConfig(adam_steps=150, lbfgs_iters=0, lr=3e-3, lr_final=1e-3, eval_every=50)
    res = train(model, problem, cfg, "cpu")
    assert all(np.isfinite(h["loss"]) for h in res.history)
    assert res.final["error"] < 0.5 * res.history[0]["error"]


CORNERS = [(m, p, t) for m in (0.0, 0.06) for p in (0.3, 0.5) for t in (0.08, 0.18)]


@pytest.mark.parametrize("m,p,t", CORNERS)
def test_joukowski_slit_lies_inside_every_corner_section(m, p, t):
    # rho = ln(|w|/R) is 0 on the slit and > 0 off it; if the slit left the section, some fluid
    # point would sit on (or across) it with rho ~ 0 and the map would be discontinuous there
    from pinncfd.pinn_airfoil_flowfield.model import joukowski_map

    rng = np.random.default_rng(0)
    pts = exterior_points(m, p, t, 4000, rng)
    q = torch.tensor([[m, p, t, 0.0]] * len(pts), dtype=torch.float64)
    rho, _ = joukowski_map(torch.tensor(pts), q)
    surf = contour(m, p, t, 400)[1:-1]  # skip the trailing edge, the slit's end point
    rho_s, _ = joukowski_map(torch.tensor(surf), q[: len(surf)])
    assert rho.min() > 0 and rho_s.min() > 0


def test_joukowski_log_radius_is_harmonic_and_tends_to_ln_r():
    from pinncfd.pinn_airfoil_flowfield.model import joukowski_map

    pts = exterior_points(0.04, 0.4, 0.12, 500, np.random.default_rng(2))
    xy = torch.tensor(pts, requires_grad=True)
    q = torch.tensor([[0.04, 0.4, 0.12, 0.0]] * len(pts), dtype=torch.float64)
    rho, _ = joukowski_map(xy, q)
    # ln|w(z)| is harmonic because w is analytic and non-zero outside the slit
    assert torch.allclose(laplacian(rho, xy), torch.zeros_like(rho), atol=1e-9)
    # far away w ~ z - c, so rho - ln r -> -ln R, a constant (here R ~ (1 - 0.55 t^2)/4)
    far = torch.tensor([[50.0, 0.0], [0.5, 50.0], [-49.5, 0.0], [0.5, -50.0]], dtype=torch.float64)
    rf, _ = joukowski_map(far, q[:4])
    r = torch.hypot(far[:, 0] - 0.5, far[:, 1])
    shift = (rf.ravel() - torch.log(r)).numpy()
    assert np.ptp(shift) < 1e-3


def test_plain_and_joukowski_models_build_and_evaluate():
    xy = torch.tensor([[0.3, 0.2], [2.0, -1.0]], requires_grad=True)
    q = torch.tensor([[0.02, 0.4, 0.12, 0.05]] * 2)
    for features in ("joukowski", "compress"):
        m = AirfoilPINN(mlp=dict(width=8, depth=1), head=dict(width=4, depth=1), features=features)
        u, v = m.velocity(xy, q)
        assert u.shape == (2, 1) and torch.isfinite(u).all() and torch.isfinite(v).all()
    with pytest.raises(ValueError):
        AirfoilPINN(features="polar")
