"""RES-01 inverse problem: taps, readings, the model-based oracle and the air-data PINN."""

import math

import numpy as np
import pytest
import torch

from pinncfd import benchmark as B
from pinncfd.inverse_airdata import AirDataPINN, AirDataProblem, oracle, taps
from pinncfd.neuroevolution_search import genome as G
from pinncfd.pinn_joukowski import JoukowskiFlow
from pinncfd.pinn_joukowski.model import exact_stream_function

FLOW = JoukowskiFlow()
ALPHA, U = np.radians(4.0), 1.15
SMALL = dict(n_interior=256, n_body=64, n_far=64, n_eval=256, n_eval_body=100)


@pytest.mark.parametrize("n", taps.LAYOUTS)
def test_taps_lie_on_the_body_around_the_nose_symmetrically(n):
    p = taps.tap_points(FLOW, n)
    zeta = FLOW.to_circle(p[:, 0], p[:, 1])
    assert np.allclose(np.abs(zeta - FLOW.zeta0), FLOW.a, atol=1e-9)  # on the body
    assert np.allclose(taps.tap_angles(n), -taps.tap_angles(n)[::-1])  # symmetric about the LE
    assert p[:, 0].max() < 0.3  # all on the forward part of the section, as FADS ports are


@pytest.mark.parametrize("alpha", np.radians([0.0, 4.0, 8.0]))
def test_clean_readings_are_the_exact_pressure(alpha):
    n = 8
    p = taps.tap_points(FLOW, n)
    # Cp from the circle plane equals the field formula at the mapped points
    assert np.allclose(taps.exact_cp(FLOW, n, alpha), FLOW.cp(p[:, 0], p[:, 1], alpha), atol=1e-9)
    dp = taps.readings(FLOW, n, alpha, U, 0.0, "val", 0)
    assert np.allclose(dp, 0.5 * U**2 * taps.exact_cp(FLOW, n, alpha))


def test_noise_has_the_stated_size_and_disjoint_reproducible_draws():
    n, level, q = 10, 0.02, 0.5 * U**2
    clean = taps.readings(FLOW, n, ALPHA, U, 0.0, "val", 0)
    noise = np.concatenate(
        [taps.readings(FLOW, n, ALPHA, U, level, "val", d) - clean for d in range(500)]
    )
    # 5000 samples: the sample std is within 5 % of sigma with overwhelming probability
    assert noise.std() == pytest.approx(level * q, rel=0.05)
    a = taps.readings(FLOW, n, ALPHA, U, level, "val", 3)
    assert np.array_equal(a, taps.readings(FLOW, n, ALPHA, U, level, "val", 3))
    assert not np.allclose(a, taps.readings(FLOW, n, ALPHA, U, level, "test", 3))


@pytest.mark.parametrize("n", taps.LAYOUTS)
@pytest.mark.parametrize("alpha", np.radians([0.0, 4.0, 8.0]))
def test_the_oracle_is_exact_without_noise(n, alpha):
    dp = taps.readings(FLOW, n, alpha, U, 0.0, "val", 0)
    a, u = oracle.fit(FLOW, n, dp)
    assert a == pytest.approx(alpha, abs=1e-10) and u == pytest.approx(U, rel=1e-10)


def test_the_exact_flow_fits_the_readings_and_scores_zero():
    class Exact(AirDataPINN):
        shift = 0.0

        def perturbation(self, xy):
            psi = exact_stream_function(xy, self.section, float(self.alpha_hat.detach()))
            return psi - self.known_part(xy) - self.shift

    m = Exact(ALPHA, U, "plain").double()
    with torch.no_grad():  # set again in float64 (the parameters were created in float32)
        m.alpha_hat.fill_(ALPHA)
        m.log_u.fill_(math.log(U))
        m.gamma.fill_(FLOW.circulation(ALPHA))
    p = AirDataProblem(m, ALPHA, U, n_taps=8, dtype=torch.float64, **SMALL)
    m.shift = float(m.perturbation(p.xy_far).mean().detach())
    with torch.no_grad():
        m.psi_b.fill_(U * (FLOW.body_value(ALPHA) - m.shift))
    losses = {k: float(v.detach()) for k, v in p.losses(m).items()}
    # taps sit on the body, where the velocity comes through the map's derivative near its
    # stagnation region: float64 round-off of ~1e-8 in the velocity leaves ~1e-16 squared
    assert losses["taps"] < 1e-14 and losses["body"] < 1e-9 and losses["kutta"] < 1e-5
    out = p.evaluate(m)
    assert out["error"] < 1e-12 and out["u_err"] < 1e-12 and out["field_error"] < 1e-12


def test_benchmark_builds_and_runs_the_airdata_case():
    budget = dict(adam_steps=30, lbfgs_iters=0, eval_every=10, resample_every=10)
    kwargs = dict(noise=0.02, n_interior=128, n_eval=128)
    out = B.run("airdata", G.AIRDATA_HAND_GENOME, budget, 0, torch.device("cpu"), kwargs)
    assert {"alpha_hat", "u_hat", "gamma", "psi_b"} <= set(out["scalars"])
    assert math.isfinite(out["val"]["error"]) and "u_err" in out["test"]
    first, last = out["result"].history[0], out["result"].history[-1]
    assert last["loss"] < first["loss"]  # the joint fit makes progress
