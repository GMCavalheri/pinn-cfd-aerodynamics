"""Shared core: networks, autodiff, sampling, convergence bookkeeping and the budget record."""

import numpy as np
import pytest
import torch

from pinncfd import io, sampling
from pinncfd.autodiff import grad, laplacian
from pinncfd.nn import MLP, n_parameters
from pinncfd.training import plateau_step, steps_to_target


def _xy(n=64, seed=0, lo=-1.0, hi=1.0):
    g = torch.Generator().manual_seed(seed)
    return (lo + (hi - lo) * torch.rand(n, 2, generator=g, dtype=torch.float64)).requires_grad_()


@pytest.mark.parametrize(
    "f",
    [
        lambda x, y: x**2 - y**2,  # Re z^2
        lambda x, y: torch.exp(x) * torch.sin(y),  # Im e^z
        lambda x, y: torch.log(x**2 + y**2),  # 2 Re log z, harmonic off the origin
    ],
)
def test_laplacian_of_harmonic_functions_vanishes(f):
    xy = _xy(lo=0.5, hi=2.0)
    u = f(xy[:, :1], xy[:, 1:])
    assert torch.allclose(laplacian(u, xy), torch.zeros_like(u), atol=1e-12)


def test_laplacian_and_gradient_of_a_quadratic():
    # u = x^2 + 3 x y + 2 y^2: grad = (2x + 3y, 3x + 4y), Laplacian = 2 + 4
    xy = _xy()
    x, y = xy[:, :1], xy[:, 1:]
    u = x**2 + 3 * x * y + 2 * y**2
    g = grad(u, xy)
    assert torch.allclose(g[:, :1], 2 * x + 3 * y)
    assert torch.allclose(g[:, 1:], 3 * x + 4 * y)
    assert torch.allclose(laplacian(u, xy), torch.full_like(u, 6.0))


@pytest.mark.parametrize("activation", ["tanh", "sin", "silu"])
def test_mlp_shapes_and_parameter_count(activation):
    m = MLP(in_dim=2, out_dim=3, width=20, depth=3, activation=activation)
    assert m(torch.zeros(5, 2)).shape == (5, 3)
    assert n_parameters(m) == (2 * 20 + 20) + 2 * (20 * 20 + 20) + (20 * 3 + 3)


def test_mlp_fourier_features_and_input_map():
    m = MLP(in_dim=2, width=16, depth=2, fourier_features=8, fourier_scale=2.0, input_scale=[5, 5])
    assert m.features.out_dim == 16
    assert m(torch.ones(4, 2)).shape == (4, 1)
    # same seed -> same network
    m2 = MLP(**m.config)
    x = torch.randn(3, 2)
    assert torch.equal(m(x), m2(x))


def test_annulus_sampler_stays_in_the_domain_and_clusters_near_the_body():
    rng = np.random.default_rng(0)
    p = sampling.annulus(20000, 1.0, 5.0, rng)
    r = np.hypot(*p.T)
    assert r.min() >= 1.0 and r.max() <= 5.0
    # area-uniform alone would put (1.5^2 - 1)/(25 - 1) = 5.2 % of points within 0.5 of the
    # body; the clustered half adds (0.5/4)^(1/3) = 50 % of its points there
    near = np.mean(r < 1.5)
    assert near == pytest.approx(0.5 * 0.052 + 0.5 * 0.5, abs=0.01)


def test_polar_grid_spans_the_annulus():
    p = sampling.polar_grid(10, 16, 1.0, 5.0)
    r = np.hypot(*p.T)
    assert p.shape == (160, 2)
    assert r.min() == pytest.approx(1.0) and r.max() == pytest.approx(5.0)


def test_steps_to_target_and_plateau():
    steps = np.arange(0, 1100, 100)
    errors = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.03, 0.0299, 0.0298, 0.0298, 0.0298, 0.0298])
    assert steps_to_target(steps, errors, 0.1) == 300
    assert steps_to_target(steps, errors, 1e-3) is None
    # from step 600 on, the best error improves by < 1 % over any 200-step window;
    # step 500 -> 700 still improves by 0.67 %, but 400 -> 600 improves by 40 %
    assert plateau_step(steps, errors, window=200, tol=0.01) == 500
    # a curve that never stops improving plateaus at its last point
    assert plateau_step(steps, 0.5 ** (steps / 100), window=200, tol=0.01) == 1000


def test_record_run_replaces_the_same_run(tmp_path):
    path = tmp_path / "budget.csv"
    base = {"phase": "p1", "case": "c", "seed": 0, "error_name": "error", "error": 0.1}
    io.record_run(base, path)
    io.record_run({**base, "seed": 1}, path)
    io.record_run({**base, "error": 0.05}, path)
    rows = io.read_budget(path)
    assert [(r["seed"], r["error"]) for r in rows] == [("0", "0.05"), ("1", "0.1")]
    with pytest.raises(KeyError):
        io.record_run({**base, "bogus": 1}, path)


def test_checkpoint_round_trip(tmp_path):
    m = MLP(in_dim=2, width=8, depth=2, seed=3)
    with torch.no_grad():
        m.linears[0].bias.fill_(0.25)
    path = io.save_checkpoint(m, tmp_path / "m.pt", {"note": "x"})
    m2, meta = io.load_checkpoint(path)
    x = torch.randn(5, 2)
    assert torch.equal(m(x), m2(x)) and meta == {"note": "x"}
