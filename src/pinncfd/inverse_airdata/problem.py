"""Loss terms and metrics for the air-data inverse problem.

The physics terms are the Joukowski case's (``pinn_joukowski.problem``). One data term is
added:

* ``taps`` — misfit of the predicted dp = (1/2)(U^2 - |V|^2) at the taps to the readings,
  divided by the largest reading (about the stagnation pressure, so the term is O(1) whatever
  the true speed, without using it).

Metrics: ``error`` is the incidence error |alpha_hat - alpha| in degrees (the primary result);
``u_err`` the relative speed error; ``field_error`` the velocity error against the exact flow
at the true condition (a diagnostic of the learned flow, on the validation or test points).
"""

from __future__ import annotations

import math

import numpy as np
import torch

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
        w = {"taps": 10.0} | dict(weights or {})
        super().__init__(model, weights=w, **kw)
        model.alpha = model_alpha
        self.weights["taps"] = w["taps"]
        self.xy_taps = self._t(self.tap_xy, True)
        self.dp_t = self._t(self.dp[:, None])

    @property
    def n_collocation(self) -> int:
        return super().n_collocation + self.n_taps

    def losses(self, model) -> dict[str, torch.Tensor]:
        out = self.physics_losses(model)
        u, v = model.velocity(self.xy_taps)
        dp = 0.5 * (model.u**2 - u**2 - v**2)
        out["taps"] = ((dp - self.dp_t) / self.dp_scale).pow(2).mean()
        return out

    def evaluate(self, model, split: str | None = None) -> dict[str, float]:
        ref = self.refs[split or self.split]
        u, v = self.predict(model, ref["field_xy"])
        uv_ref = ref["field_uv"] * self.u_true
        duv = np.column_stack([u, v]) - uv_ref
        a_hat = float(model.alpha_hat.detach())
        u_hat = float(model.u.detach())
        return {
            "error": math.degrees(abs(a_hat - self.alpha_true)),
            "u_err": abs(u_hat - self.u_true) / self.u_true,
            "field_error": float(np.linalg.norm(duv) / np.linalg.norm(uv_ref)),
            "alpha_hat_deg": math.degrees(a_hat),
            "u_hat": u_hat,
        }
