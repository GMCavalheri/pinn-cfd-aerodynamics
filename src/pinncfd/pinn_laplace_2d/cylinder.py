"""PINNs for potential flow past a circular cylinder.

The flow is solved on the annulus 1 <= r <= R (R = ``r_far``) with Dirichlet data on the
outer circle. Four formulations are provided, chosen with ``formulation`` and ``hard``:

* ``"psi"``, soft constraints: the network gives the stream function; the loss penalises the
  Laplace residual, psi on the body (psi = 0, the body is a streamline) and psi on the outer
  circle.
* ``"psi"``, hard constraints: psi = psi_b + d N with d = (r - 1)(R - r) / ((R - 1)/2)^2
  vanishing on both circles and psi_b = psi_R(theta) (r - 1)/(R - 1) matching both boundary
  conditions exactly, so only the Laplace residual is left in the loss (Lagaris et al. 1998;
  Sukumar & Srivastava 2022).
* ``"phi"``, soft constraints: the network gives the velocity potential; the body condition
  is the Neumann condition d(phi)/dn = 0. Only for Gamma = 0: with circulation phi is
  multivalued (it jumps by Gamma around the body), which is why the project works with psi.

The outer Dirichlet data comes from the exact solution: this phase is a verification case,
and the question is how well the network solves the boundary-value problem, not how well a
truncated far field approximates infinity (that question returns with the airfoils).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from pinncfd import sampling
from pinncfd.autodiff import grad, laplacian
from pinncfd.nn import MLP
from pinncfd.pinn_laplace_2d import exact


class CylinderPINN(nn.Module):
    """A network plus the formulation that turns its output into psi or phi."""

    def __init__(
        self,
        formulation: str = "psi",
        hard: bool = True,
        gamma: float = 0.0,
        r_far: float = 5.0,
        mlp: dict | None = None,
    ):
        super().__init__()
        if formulation not in ("psi", "phi"):
            raise ValueError("formulation must be 'psi' or 'phi'")
        if formulation == "phi" and (hard or gamma):
            raise ValueError("the phi formulation is implemented soft and non-lifting only")
        mlp = dict(mlp or {})
        mlp.setdefault("in_dim", 2)
        mlp.setdefault("input_scale", [r_far, r_far])
        self.net = MLP(**mlp)
        self.formulation, self.hard, self.gamma, self.r_far = formulation, hard, gamma, r_far
        self.config = dict(
            formulation=formulation, hard=hard, gamma=gamma, r_far=r_far, mlp=self.net.config
        )

    def outer_value(self, x, y, r):
        """Dirichlet data psi_R(theta) (or phi_R) on the outer circle, as a field of theta."""
        R = self.r_far
        if self.formulation == "psi":
            return (R - 1.0 / R) * y / r + self.gamma / (2 * math.pi) * math.log(R)
        return (R + 1.0 / R) * x / r

    def forward(self, xy):
        n = self.net(xy)
        if not self.hard:
            return n
        x, y = xy[:, :1], xy[:, 1:2]
        r = torch.sqrt(x**2 + y**2)
        R = self.r_far
        d = (r - 1.0) * (R - r) / ((R - 1.0) / 2) ** 2
        return self.outer_value(x, y, r) * (r - 1.0) / (R - 1.0) + d * n

    def velocity(self, xy):
        """(u, v) from the derivatives of psi or phi; ``xy`` must require grad."""
        g = grad(self(xy), xy)
        if self.formulation == "psi":
            return g[:, 1:2], -g[:, 0:1]
        return g[:, 0:1], g[:, 1:2]


class CylinderProblem:
    """Loss terms, collocation points and error metrics for :class:`CylinderPINN`."""

    def __init__(
        self,
        model: CylinderPINN,
        n_interior: int = 8192,
        n_boundary: int = 512,
        weights: dict | None = None,
        n_eval_r: int = 64,
        n_eval_theta: int = 128,
        n_surface: int = 360,
        device=None,
        seed: int = 0,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.n_interior, self.n_boundary = n_interior, n_boundary
        terms = ["pde"] if model.hard else ["pde", "body", "far"]
        self.weights = {k: 1.0 for k in terms}
        self.weights.update(weights or {})
        self.resample(np.random.default_rng(seed))

        # fixed evaluation set (float64 references)
        R = model.r_far
        self.eval_xy = sampling.polar_grid(n_eval_r, n_eval_theta, 1.0, R)
        ex, ey = self.eval_xy.T
        self.u_ref = np.column_stack(exact.velocity(ex, ey, model.gamma))
        self.f_ref = (
            exact.stream_function(ex, ey, model.gamma)
            if model.formulation == "psi"
            else exact.potential(ex, ey)
        )
        self.theta = np.linspace(0, 2 * np.pi, n_surface, endpoint=False)
        self.surf_xy = np.column_stack([np.cos(self.theta), np.sin(self.theta)])
        self.cp_ref = exact.surface_cp(self.theta, model.gamma)

    @property
    def n_collocation(self) -> int:
        return self.n_interior + 2 * self.n_boundary

    def _t(self, a, requires_grad=False):
        return torch.tensor(a, dtype=torch.float32, device=self.device).requires_grad_(
            requires_grad
        )

    def resample(self, rng: np.random.Generator) -> None:
        R = self.model.r_far
        self.xy_in = self._t(sampling.annulus(self.n_interior, 1.0, R, rng), True)
        self.xy_body = self._t(sampling.circle(self.n_boundary, 1.0, rng), True)
        self.xy_far = self._t(sampling.circle(self.n_boundary, R, rng))
        fx, fy = self.xy_far[:, :1], self.xy_far[:, 1:2]
        self.f_far = self.model.outer_value(fx, fy, torch.sqrt(fx**2 + fy**2)).detach()

    def losses(self, model) -> dict[str, torch.Tensor]:
        out = {"pde": laplacian(model(self.xy_in), self.xy_in).pow(2).mean()}
        if model.hard:
            return out
        if model.formulation == "psi":
            out["body"] = model(self.xy_body).pow(2).mean()
        else:  # Neumann: n . grad(phi) = 0 with n = (x, y) on the unit circle
            g = grad(model(self.xy_body), self.xy_body)
            out["body"] = (g * self.xy_body).sum(dim=1).pow(2).mean()
        out["far"] = (model(self.xy_far) - self.f_far).pow(2).mean()
        return out

    def predict(self, model, xy: np.ndarray):
        """(f, u, v) at ``xy`` as float64 arrays (f = psi or phi)."""
        t = self._t(xy, True)
        f = model(t)
        u, v = model.velocity(t)
        return tuple(a.detach().cpu().double().numpy().ravel() for a in (f, u, v))

    def evaluate(self, model) -> dict[str, float]:
        f, u, v = self.predict(model, self.eval_xy)
        uv = np.column_stack([u, v])
        err_u = np.linalg.norm(uv - self.u_ref) / np.linalg.norm(self.u_ref)
        err_f = np.linalg.norm(f - self.f_ref) / np.linalg.norm(self.f_ref)
        _, us, vs = self.predict(model, self.surf_xy)
        cp = 1.0 - (us**2 + vs**2)
        cl = exact.cl_from_cp(self.theta, cp)
        return {
            "error": float(err_u),
            "err_f": float(err_f),
            "cp_max_abs": float(np.abs(cp - self.cp_ref).max()),
            "cp_rms": float(np.sqrt(np.mean((cp - self.cp_ref) ** 2))),
            "cl": cl,
            "cl_abs_err": abs(cl - exact.cl(model.gamma)),
        }
