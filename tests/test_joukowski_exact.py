"""RES-01 case 2: Joukowski section geometry and the exact flow field.

Every check is a property of the classical solution (Anderson, *Fundamentals of
Aerodynamics*, Sec. 4.14; Katz & Plotkin, Sec. 6.6) or AERO-01's own exact surface pressure.
"""

import numpy as np
import pytest

from pinncfd.pinn_joukowski import JoukowskiFlow

FLOW = JoukowskiFlow()
SEC = FLOW.section
ALPHAS = np.radians([0.0, 4.0, 8.0])


def ring(n=2048, r=2.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return 0.5 + r * np.cos(th), r * np.sin(th), th


def test_section_is_normalised_and_closed():
    c = SEC.contour(4000)
    assert c[0] == pytest.approx([1.0, 0.0]) and c[-1] == pytest.approx([1.0, 0.0])
    assert c[:, 0].min() == pytest.approx(0.0, abs=1e-6) and c[:, 0].max() == pytest.approx(1.0)
    # lower surface first: clockwise from the trailing edge (a quarter of the way round is
    # mid-chord on the lower surface, below the chord line; the cambered cusp itself is not)
    assert c[1000, 1] < 0 < c[3000, 1]
    _, yu, _, yl = SEC.surfaces(np.linspace(0.02, 0.98, 49))
    assert np.all(yu > yl)
    cx, cy = SEC.center()
    assert SEC.inside(np.array([[cx, cy]]))[0] and not SEC.inside(np.array([[0.5, 0.5]]))[0]


def test_surface_samples_lie_on_the_contour_with_outward_normals():
    pts, nrm = SEC.surface_samples(500, np.random.default_rng(0))
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)
    # a short step along the normal leaves the body, a step against it enters
    assert not SEC.inside(pts + 2e-4 * nrm).any()
    inner = SEC.inside(pts - 2e-4 * nrm)
    assert inner.mean() > 0.95  # the few misses sit at the cusp, thinner than the step


@pytest.mark.parametrize("alpha", ALPHAS)
def test_body_is_a_streamline(alpha):
    x, y, _ = FLOW.body_points(720, skip_te=0.02)
    psi = FLOW.stream_function(x, y, alpha)
    assert np.allclose(psi, FLOW.body_value(alpha), atol=1e-12)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_surface_cp_matches_aero01(alpha):
    # AERO-01's exact Joukowski Cp (validated against its panel method) at the same points
    x, y, _ = FLOW.body_points(720, skip_te=0.02)
    cp_aero = FLOW.foil.surface_cp(np.column_stack([x, y]), alpha)
    assert np.allclose(FLOW.cp(x, y, alpha), cp_aero, atol=1e-10)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_velocity_is_the_curl_of_psi_and_psi_is_harmonic(alpha):
    rng = np.random.default_rng(1)
    r = 0.6 + 3 * rng.random(200)
    th = 2 * np.pi * rng.random(200)
    x, y = 0.5 + r * np.cos(th), r * np.sin(th)  # all at least 0.1 chord outside the body
    h = 1e-3
    p = lambda dx, dy: FLOW.stream_function(x + dx, y + dy, alpha)  # noqa: E731
    u, v = FLOW.velocity(x, y, alpha)
    assert np.allclose((p(0, h) - p(0, -h)) / (2 * h), u, atol=1e-6)
    assert np.allclose(-(p(h, 0) - p(-h, 0)) / (2 * h), v, atol=1e-6)
    lap = (p(h, 0) + p(-h, 0) + p(0, h) + p(0, -h) - 4 * p(0, 0)) / h**2
    # five-point stencil: truncation error h^2/12 |psi''''| is below 1e-4 this far out
    assert np.abs(lap).max() < 1e-4


@pytest.mark.parametrize("alpha", ALPHAS)
def test_circulation_on_a_ring_gives_kutta_joukowski_lift(alpha):
    x, y, th = ring()
    u, v = FLOW.velocity(x, y, alpha)
    # clockwise circulation = minus the counter-clockwise line integral; trapezoidal rule on a
    # circle is spectrally accurate for this smooth integrand
    gamma_cw = -np.mean(-u * np.sin(th) + v * np.cos(th)) * 2 * np.pi * 2.0
    assert gamma_cw == pytest.approx(FLOW.circulation(alpha), rel=1e-10)
    assert 2 * gamma_cw == pytest.approx(FLOW.cl(alpha), rel=1e-10)
    # Kutta: Gamma = 4 pi a sin(alpha + beta), over the chord
    expected = 4 * np.pi * FLOW.a * np.sin(alpha + FLOW.foil.beta) / FLOW.chord
    assert FLOW.circulation(alpha) == pytest.approx(expected)


def test_far_field_is_the_freestream():
    alpha = np.radians(4)
    x, y, _ = ring(64, 2000.0)
    u, v = FLOW.velocity(x, y, alpha)
    # disturbance decays like Gamma / (2 pi r): about 1e-4 at r = 2000 chords
    assert np.allclose(u, np.cos(alpha), atol=2e-4) and np.allclose(v, np.sin(alpha), atol=2e-4)


def test_speed_stays_finite_at_the_cusped_trailing_edge():
    # Kutta condition: dF/dzeta and dz/dzeta both vanish at zeta = b, so the speed has a finite
    # limit there. Approach the cusp from downstream along the chord line.
    alpha = np.radians(4)
    d = np.logspace(-2, -7, 6)
    u, v = FLOW.velocity(1 + d, np.zeros_like(d), alpha)
    q = np.hypot(u, v)
    assert np.all(np.isfinite(q)) and q.max() < 2.0
    assert abs(q[-1] - q[-2]) < 1e-3  # converged to the trailing-edge speed
