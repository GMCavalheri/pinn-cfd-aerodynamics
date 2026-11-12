"""The air-data PINN: the Joukowski network with the incidence and speed learnable.

    psi = U [ cos(alpha) psi_x + sin(alpha) psi_y + Gamma(alpha) psi_c ],

with the three basis flows of :mod:`pinncfd.pinn_joukowski.model`, alpha (``alpha_hat``) and
log U (``log_u``) learnable, and Gamma(alpha) the Kutta circulation at the learned incidence
(the basis flows' trailing-edge jumps are detached, the dependence on alpha is not). The basis
flows do not depend on alpha or U, so the physics losses are unchanged; only the readings see
the incidence and the speed.
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
        self.alpha_hat = nn.Parameter(torch.tensor(float(alpha0)))
        self.log_u = nn.Parameter(torch.tensor(math.log(u0)))
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
