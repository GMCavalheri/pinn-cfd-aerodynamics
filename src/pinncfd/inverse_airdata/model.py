"""The air-data PINN: the Joukowski network with the incidence and speed learnable.

    psi = U [ y cos(alpha) - x sin(alpha) + Gamma / (2 pi) ln r_c + psi'(x, y) ],

with alpha (``alpha_hat``), log U (``log_u``), Gamma and psi_b learnable scalars, trained
together with psi' from the physics losses and the tap readings. U multiplies the whole stream
function, so every physics loss is unchanged in form; only the readings see the speed.
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

    def known_part(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, :1], xy[:, 1:2]
        log_r = 0.5 * torch.log((x - self.cx) ** 2 + (y - self.cy) ** 2)
        a = self.alpha_hat
        return y * torch.cos(a) - x * torch.sin(a) + self.gamma / (2 * math.pi) * log_r

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.u * (self.known_part(xy) + self.perturbation(xy))
