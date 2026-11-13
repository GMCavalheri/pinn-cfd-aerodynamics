"""Loss terms and metrics for the air-data inverse problem.

The loss terms are the Joukowski case's physics (``pinn_joukowski.problem``) and nothing else:
the basis flows do not depend on the incidence or the speed. The readings enter through
:meth:`AirDataProblem.fit_air_data`, run at every evaluation: least squares for (alpha, U) on
the misfit of the predicted dp = (1/2)(U^2 - |V|^2) at the taps, divided by the largest reading
(about the stagnation pressure, so the residuals are O(1) without using the true speed),
started every time from alpha0 and u0, as the oracle starts. A warm start from the previous fit
was tried first: an early fit to rough basis flows (55 degrees, U = 0.06) then stuck.

Metrics: ``error`` is the incidence error |alpha_hat - alpha| in degrees (the primary result);
``u_err`` the relative speed error; ``taps_rms`` the RMS of the scaled tap residuals;
``field_error`` the velocity error against the exact flow
at the true condition (a diagnostic of the learned flow, on the validation or test points).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import least_squares

from pinncfd.inverse_airdata import taps
from pinncfd.inverse_airdata.model import AirDataPINN
from pinncfd.pinn_joukowski.exact import JoukowskiFlow
from pinncfd.pinn_joukowski.problem import JoukowskiProblem


class AirDataProblem(JoukowskiProblem):
    def __init__(
        self,
        model: AirDataPINN,
        alpha: float,
        u: float,
        n_taps: int = 8,
        noise: float = 0.0,
        draw: int = 0,
        noise_split: str = "val",
        weights: dict | None = None,
        **kw,
    ):
        self.alpha_true, self.u_true = float(alpha), float(u)
        self.n_taps, self.noise, self.draw, self.noise_split = n_taps, noise, draw, noise_split
        flow = JoukowskiFlow(model.section)
        self.tap_xy = taps.tap_points(flow, n_taps)
        self.dp = taps.readings(flow, n_taps, alpha, u, noise, noise_split, draw)
        self.dp_scale = float(np.abs(self.dp).max())
        # the reference flow for field metrics is the true condition
        model_alpha = model.alpha
        model.alpha = self.alpha_true
        super().__init__(model, weights=weights, **kw)
        model.alpha = model_alpha
        self.xy_taps = self._t(self.tap_xy, True)

    @property
    def n_collocation(self) -> int:
        return super().n_collocation + self.n_taps

    def losses(self, model) -> dict[str, torch.Tensor]:
        return self.physics_losses(model)

    def tap_residuals(self, model, alpha: float, u: float, g: np.ndarray) -> np.ndarray:
        """Scaled tap misfit at (alpha, U), with ``g`` the basis velocities at the taps."""
        k = model.kutta_k
        a = torch.tensor(alpha, dtype=k.dtype, device=k.device)
        c = [torch.cos(a), torch.sin(a), model.gamma_for(a)]
        c = np.array([float(t.detach()) for t in c])
        vel = np.einsum("nkd,k->nd", g, c)  # (N, 2), unit speed
        dp = 0.5 * u**2 * (1.0 - (vel**2).sum(axis=1))
        return (dp - self.dp) / self.dp_scale

    def fit_air_data(self, model) -> np.ndarray:
        """Least-squares (alpha_hat, U) on the current basis flows; returns the residuals."""
        g = model.basis_velocity(self.xy_taps).double().cpu().numpy()
        x0 = [model.config["alpha0"], model.config["u0"]]  # as the oracle: no warm start
        sol = least_squares(
            lambda p: self.tap_residuals(model, p[0], p[1], g),
            x0,
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
        )
        if np.all(np.isfinite(sol.x)) and sol.x[1] != 0:
            with torch.no_grad():
                model.alpha_hat.fill_(float(sol.x[0]))
                model.log_u.fill_(math.log(abs(float(sol.x[1]))))
        return self.tap_residuals(model, float(model.alpha_hat), float(model.u), g)

    def evaluate(self, model, split: str | None = None) -> dict[str, float]:
        model.update_kutta()
        res = self.fit_air_data(model)
        ref = self.refs[split or self.split]
        u, v = self.predict(model, ref["field_xy"])
        uv_ref = ref["field_uv"] * self.u_true
        duv = np.column_stack([u, v]) - uv_ref
        a_hat = float(model.alpha_hat.detach())
        u_hat = float(model.u.detach())
        return {
            "error": math.degrees(abs(a_hat - self.alpha_true)),
            "u_err": abs(u_hat - self.u_true) / self.u_true,
            "taps_rms": float(np.sqrt(np.mean(res**2))),
            "field_error": float(np.linalg.norm(duv) / np.linalg.norm(uv_ref)),
            "alpha_hat_deg": math.degrees(a_hat),
            "u_hat": u_hat,
        }
