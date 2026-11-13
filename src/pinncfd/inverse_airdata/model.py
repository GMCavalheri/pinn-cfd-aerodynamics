"""The air-data PINN: the Joukowski basis flows, with the incidence and speed fitted to the taps.

    psi = U [ cos(alpha) psi_x + sin(alpha) psi_y + Gamma(alpha) psi_c ],

with the three basis flows of :mod:`pinncfd.pinn_joukowski.model`, and Gamma(alpha) the Kutta
circulation of the basis flows at the incidence ``alpha_hat``.

The basis flows do not depend on alpha or U: the physics alone determines them, and the
readings carry no information about them. So the network trains on the physics only, and
alpha_hat and U (``log_u``) are not trained by gradient steps: at every evaluation
``AirDataProblem.fit_air_data`` solves the least-squares problem for them on the current basis
flows, as the oracle does on the exact flow. They are parameters with ``requires_grad=False``,
so no optimiser touches them while checkpoints and the learned scalars still carry them.

Two earlier versions failed (pilot of 2026-11-12). With the tap misfit as a loss term that also
trained the network, the network bent the basis flows to fit the readings at the starting
incidence, alpha_hat barely moved and the incidence error was 4.2-4.5 degrees. With the basis
flows detached in that term, alpha_hat and U still chased basis flows that were far from
converged (alpha_hat drifted to -20 degrees in a short run), and the coupled objective stopped
L-BFGS after a few iterations.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pinncfd.pinn_joukowski.model import JoukowskiPINN


class AirDataPINN(JoukowskiPINN):
    def __init__(
        self,
        alpha0: float = 0.0,
        u0: float = 1.0,
        features: str = "plain",
        mlp: dict | None = None,
        mx: float = 0.1,
        my: float = 0.1,
    ):
        super().__init__(alpha0, features, mlp, mx, my)
        self.alpha_hat = nn.Parameter(torch.tensor(float(alpha0)), requires_grad=False)
        self.log_u = nn.Parameter(torch.tensor(math.log(u0)), requires_grad=False)
        self.config = dict(
            alpha0=float(alpha0),
            u0=float(u0),
            features=features,
            mlp=self.field.config,
            mx=mx,
            my=my,
        )

    @property
    def u(self) -> torch.Tensor:
        return torch.exp(self.log_u)

    def incidence(self) -> torch.Tensor:
        return self.alpha_hat

    def speed(self) -> torch.Tensor:
        return self.u

    def basis_velocity(self, xy: torch.Tensor) -> torch.Tensor:
        """Velocities of the three basis flows at ``xy``, detached: (N, 3, 2) with
        [..., 0] = d(psi_k)/dy and [..., 1] = -d(psi_k)/dx. The flow's velocity is
        U sum_k c_k of these, with c = (cos alpha, sin alpha, Gamma(alpha))."""
        xy = xy.detach().clone().requires_grad_(True)
        basis = self.basis(xy)
        grads = [
            torch.autograd.grad(basis[:, k].sum(), xy, retain_graph=k < 2)[0] for k in range(3)
        ]
        g = torch.stack(grads, dim=1).detach()  # d(psi_k)/dx, d(psi_k)/dy
        return torch.stack([g[..., 1], -g[..., 0]], dim=-1)
