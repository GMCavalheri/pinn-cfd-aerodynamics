"""PINNs for the steady 2D Euler equations in conservation form (exploratory, Phase 5).

    d F(U)/dx + d G(U)/dy = 0,   U = (rho, rho u, rho v, E),
    F = (rho u, rho u^2 + p, rho u v, u (E + p)),
    G = (rho v, rho u v, rho v^2 + p, v (E + p)),
    E = p/(gamma - 1) + rho (u^2 + v^2)/2.

The network outputs primitive variables; density and pressure pass through a softplus so they
stay positive. The conservation form matters for shocks: its weak solutions obey the
Rankine-Hugoniot jump conditions, while a non-conservative (primitive) form would admit wrong
jumps. An optional artificial viscosity nu adds nu * Laplacian(U) to the residual, smearing a
shock over a width of order nu / (velocity jump) so a smooth network can represent it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pinncfd.autodiff import grad, laplacian
from pinncfd.nn import MLP
from pinncfd.pinn_euler_2d.exact import (
    GAMMA,
    oblique_shock,
    ringleb,
    ringleb_boundary,
    ringleb_samples,
    wedge_field,
)


class EulerPINN(nn.Module):
    """(x, y) -> primitive (rho, u, v, p), scaled about a reference state."""

    def __init__(self, mlp: dict | None = None, center=(0.0, 0.0), scale=(1.0, 1.0), ref=None):
        super().__init__()
        mlp = dict(dict(width=64, depth=5, activation="tanh"), **(mlp or {}))
        mlp.update(in_dim=2, out_dim=4, input_center=list(center), input_scale=list(scale))
        self.net = MLP(**mlp)
        self.register_buffer("ref", torch.tensor(ref if ref is not None else [1.0, 1.0, 1.0, 1.0]))
        self.config = dict(
            mlp=self.net.config,
            center=list(center),
            scale=list(scale),
            ref=[float(r) for r in self.ref],
        )

    def forward(self, xy):
        o = self.net(xy)
        rho = self.ref[0] * nn.functional.softplus(o[:, 0:1] + 0.5413)  # softplus(0.5413) = 1
        u = self.ref[1] * o[:, 1:2]
        v = self.ref[2] * o[:, 2:3]
        p = self.ref[3] * nn.functional.softplus(o[:, 3:4] + 0.5413)
        return torch.cat([rho, u, v, p], dim=1)


def fluxes(w):
    rho, u, v, p = w[:, 0:1], w[:, 1:2], w[:, 2:3], w[:, 3:4]
    E = p / (GAMMA - 1) + 0.5 * rho * (u**2 + v**2)
    U = torch.cat([rho, rho * u, rho * v, E], dim=1)
    F = torch.cat([rho * u, rho * u**2 + p, rho * u * v, u * (E + p)], dim=1)
    G = torch.cat([rho * v, rho * u * v, rho * v**2 + p, v * (E + p)], dim=1)
    return U, F, G


def residual(model, xy, nu: float = 0.0):
    """(N, 4) residual dF/dx + dG/dy - nu * Laplacian(U)."""
    U, F, G = fluxes(model(xy))
    cols = []
    for i in range(4):
        r = grad(F[:, i : i + 1], xy)[:, 0:1] + grad(G[:, i : i + 1], xy)[:, 1:2]
        if nu:
            r = r - nu * laplacian(U[:, i : i + 1], xy)
        cols.append(r)
    return torch.cat(cols, dim=1)


class _EulerProblem:
    """Shared plumbing: tensors, prediction, error against an exact field."""

    weights: dict

    def _t(self, a, requires_grad=False):
        t = torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)
        return t.requires_grad_(requires_grad)

    def predict(self, model, xy: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(self._t(xy)).cpu().double().numpy()

    def state_errors(self, model, xy, ref) -> dict:
        w = self.predict(model, xy)
        out = {"error": float(np.linalg.norm(w - ref) / np.linalg.norm(ref))}
        for i, name in enumerate(("rho", "u", "v", "p")):
            scale = np.linalg.norm(ref[:, i]) or np.linalg.norm(ref)
            out[f"err_{name}"] = float(np.linalg.norm(w[:, i] - ref[:, i]) / scale)
        return out


class RinglebProblem(_EulerProblem):
    """Ringleb flow with exact Dirichlet data on the whole boundary."""

    def __init__(
        self, model, n_interior=8192, n_boundary=1024, nu=0.0, weights=None, device=None, seed=0
    ):
        self.model, self.nu = model, nu
        self.device = device or next(model.parameters()).device
        self.n_interior, self.n_boundary = n_interior, n_boundary
        self.weights = {"pde": 1.0, "bc": 10.0} | (weights or {})
        rng = np.random.default_rng(seed + 1000)
        e = ringleb_samples(4096, rng)
        x, y, *w = ringleb(*e.T)
        self.eval_xy, self.eval_w = np.column_stack([x, y]), np.column_stack(w)
        self.resample(np.random.default_rng(seed))

    @property
    def n_collocation(self):
        return self.n_interior + self.n_boundary

    def resample(self, rng):
        qkb = ringleb_samples(self.n_interior, rng)
        x, y, *_ = ringleb(*qkb.T)
        self.xy_in = self._t(np.column_stack([x, y]), True)
        qkb = ringleb_boundary(self.n_boundary, rng)
        x, y, *w = ringleb(*qkb.T)
        self.xy_bc, self.w_bc = self._t(np.column_stack([x, y])), self._t(np.column_stack(w))

    def losses(self, model):
        return {
            "pde": residual(model, self.xy_in, self.nu).pow(2).mean(),
            "bc": (model(self.xy_bc) - self.w_bc).pow(2).mean(),
        }

    def evaluate(self, model):
        return self.state_errors(model, self.eval_xy, self.eval_w)


class WedgeProblem(_EulerProblem):
    """Mach-M flow over a wedge of half-angle theta in [0, 1] x [x tan(theta), 1].

    Supersonic inflow (x = 0) and top (y = 1, which the shock does not reach for M = 2,
    theta = 10 deg): freestream Dirichlet. Wall: slip, u . n = 0. Outflow (x = 1): supersonic,
    no condition.
    """

    def __init__(
        self,
        model,
        mach=2.0,
        theta_deg=10.0,
        n_interior=8192,
        n_boundary=1024,
        nu=0.0,
        weights=None,
        device=None,
        seed=0,
    ):
        self.model, self.nu = model, nu
        self.mach, self.theta = mach, np.radians(theta_deg)
        self.device = device or next(model.parameters()).device
        self.n_interior, self.n_boundary = n_interior, n_boundary
        self.shock = oblique_shock(self.theta, mach)
        self.weights = {"pde": 1.0, "inflow": 10.0, "wall": 10.0} | (weights or {})
        g = np.linspace(0, 1, 101)
        xx, yy = np.meshgrid(g, g)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        self.eval_xy = pts[pts[:, 1] >= pts[:, 0] * np.tan(self.theta)]
        self.eval_w = wedge_field(*self.eval_xy.T, self.theta, mach)
        self.resample(np.random.default_rng(seed))

    @property
    def n_collocation(self):
        return self.n_interior + self.n_boundary

    def resample(self, rng):
        tan = np.tan(self.theta)
        pts = rng.random((2 * self.n_interior, 2))
        pts = pts[pts[:, 1] > pts[:, 0] * tan][: self.n_interior]
        self.xy_in = self._t(pts, True)
        n2 = self.n_boundary // 2
        left = np.column_stack([np.zeros(n2 // 2), rng.random(n2 // 2)])
        top = np.column_stack([rng.random(n2 - n2 // 2), np.ones(n2 - n2 // 2)])
        self.xy_in_bc = self._t(np.vstack([left, top]))
        self.w_in_bc = self._t(np.tile(self.shock["state1"], (n2, 1)))
        xw = rng.random(self.n_boundary - n2)
        self.xy_wall = self._t(np.column_stack([xw, xw * tan]))
        self.n_wall = self._t(np.array([-np.sin(self.theta), np.cos(self.theta)]))

    def losses(self, model):
        w = model(self.xy_wall)
        return {
            "pde": residual(model, self.xy_in, self.nu).pow(2).mean(),
            "inflow": (model(self.xy_in_bc) - self.w_in_bc).pow(2).mean(),
            "wall": (w[:, 1:3] @ self.n_wall).pow(2).mean(),
        }

    def shock_angle(self, model, xs=(0.4, 0.6, 0.8, 1.0)) -> float:
        """Shock angle [rad] from where the density crosses its mid value on vertical lines."""
        mid = 0.5 * (self.shock["state1"][0] + self.shock["state2"][0])
        angles = []
        for x0 in xs:
            y = np.linspace(x0 * np.tan(self.theta), 1, 2001)
            rho = self.predict(model, np.column_stack([np.full_like(y, x0), y]))[:, 0]
            cross = np.flatnonzero((rho[:-1] - mid) * (rho[1:] - mid) <= 0)
            if len(cross):
                angles.append(np.arctan2(y[cross[-1]], x0))
        return float(np.mean(angles)) if angles else float("nan")

    def evaluate(self, model):
        out = self.state_errors(model, self.eval_xy, self.eval_w)
        beta = self.shock_angle(model)
        # errors away from the shock (outside a band of +-0.05 in y around it)
        x, y = self.eval_xy.T
        far = np.abs(y - x * np.tan(self.shock["beta"])) > 0.05
        out["error_smooth"] = self.state_errors(model, self.eval_xy[far], self.eval_w[far])["error"]
        out["beta_deg"] = float(np.degrees(beta))
        out["beta_err_deg"] = float(abs(np.degrees(beta - self.shock["beta"])))
        return out
