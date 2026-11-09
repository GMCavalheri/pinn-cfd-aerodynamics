"""A PINN for the flow around one Joukowski section at one incidence (RES-01 case 2).

The stream function is split as in the airfoil-family network (``pinn_airfoil_flowfield``),

    psi(x, y) = y cos(alpha) - x sin(alpha) + Gamma / (2 pi) ln r_c + psi'(x, y),

with r_c the distance to a point inside the section (the camber line at mid-chord). Gamma and
the body value psi_b are learnable scalars: one network per section and incidence, so no case
head is needed. psi' is the network, and the question RES-01 asks of this case is how well it
can be trained when it sees the plane through ordinary coordinates. ``features`` chooses them:

* ``"plain"`` — (x - 0.5, y) in chords: the coordinates themselves;
* ``"compress"`` — the radially compressed coordinates of the airfoil family
  (``pinn_airfoil_flowfield.model.compress``), which shrink the far field;
* ``"exact_map"`` — **reference only**: log-polar coordinates of the true circle plane of this
  section, (ln(|zeta'|/a), cos, sin of arg zeta'). There the exact solution is a few smooth
  harmonics, so these features hand the network most of the answer. They measure that gift,
  and are never offered to a search.

Fourier features, when the genome asks for them, are added inside the MLP on top of these.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pinncfd.external.geometry import JoukowskiSection
from pinncfd.nn import MLP
from pinncfd.pinn_airfoil_flowfield.model import X0, compress

FEATURES = ("plain", "compress", "exact_map")


def circle_plane(xy: torch.Tensor, section: JoukowskiSection):
    """zeta' = zeta - zeta0 of normalised points, as a complex tensor (differentiable).

    Torch port of :meth:`JoukowskiFlow.to_circle`: z = zeta + b^2/zeta inverted, keeping the
    root outside the generating circle.
    """
    foil = section.foil
    b, z0 = foil.b, foil.zeta0
    x = xy[:, 0:1] * section.chord + section.x_le
    y = xy[:, 1:2] * section.chord
    z = torch.complex(x, y)
    s = torch.sqrt(z * z - 4 * b * b)
    zp, zm = (z + s) / 2 - z0, (z - s) / 2 - z0
    return torch.where(zp.abs() >= zm.abs(), zp, zm)


def exact_map_features(xy: torch.Tensor, section: JoukowskiSection) -> torch.Tensor:
    w = circle_plane(xy, section)
    rho = torch.log(w.abs() / section.foil.radius)
    unit = w / w.abs()
    return torch.cat([rho, unit.real, unit.imag], dim=1)


class JoukowskiPINN(nn.Module):
    def __init__(
        self,
        alpha: float,
        features: str = "plain",
        mlp: dict | None = None,
        mx: float = 0.1,
        my: float = 0.1,
    ):
        super().__init__()
        if features not in FEATURES:
            raise ValueError(f"features must be one of {FEATURES}")
        self.section = JoukowskiSection(mx, my)
        self.alpha, self.features = float(alpha), features
        self.cx, self.cy = self.section.center()
        if features == "exact_map":  # rho in [0, ~2.5], cos, sin
            center, scale = [1.25, 0.0, 0.0], [1.25, 1.0, 1.0]
        else:
            center, scale = [0.0, 0.0], [1.0, 1.0]
        mlp = dict(dict(width=64, depth=5, activation="tanh"), **(mlp or {}))
        mlp.update(in_dim=len(center), out_dim=1, input_center=center, input_scale=scale)
        self.field = MLP(**mlp)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.psi_b = nn.Parameter(torch.zeros(()))
        self.config = dict(
            alpha=self.alpha,
            features=features,
            mlp=self.field.config,
            mx=self.section.mx,
            my=self.section.my,
        )

    def inputs(self, xy: torch.Tensor) -> torch.Tensor:
        if self.features == "plain":
            return torch.cat([xy[:, :1] - X0, xy[:, 1:2]], dim=1)
        if self.features == "compress":
            return compress(xy)
        return exact_map_features(xy, self.section)

    def perturbation(self, xy: torch.Tensor) -> torch.Tensor:
        return self.field(self.inputs(xy))

    def known_part(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, :1], xy[:, 1:2]
        log_r = 0.5 * torch.log((x - self.cx) ** 2 + (y - self.cy) ** 2)
        a = self.alpha
        return y * math.cos(a) - x * math.sin(a) + self.gamma / (2 * math.pi) * log_r

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.known_part(xy) + self.perturbation(xy)

    def velocity(self, xy: torch.Tensor):
        """(u, v) = (d psi/dy, -d psi/dx); ``xy`` must require grad."""
        (g,) = torch.autograd.grad(
            self(xy), xy, grad_outputs=torch.ones(len(xy), 1, device=xy.device), create_graph=True
        )
        return g[:, 1:2], -g[:, 0:1]


def exact_stream_function(xy: torch.Tensor, section: JoukowskiSection, alpha: float):
    """The exact psi of :class:`JoukowskiFlow` in torch (differentiable), for loss checks."""
    foil = section.foil
    w = circle_plane(xy, section)
    g = foil.circulation(alpha)
    f = (
        w * complex(math.cos(alpha), -math.sin(alpha))
        + foil.radius**2 * complex(math.cos(alpha), math.sin(alpha)) / w
    )
    return (f.imag + g / (2 * math.pi) * torch.log(w.abs())) / section.chord
