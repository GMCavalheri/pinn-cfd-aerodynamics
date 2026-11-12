"""A PINN for the flow around one Joukowski section at one incidence (RES-01 case 2).

**Superposition.** Potential flow is linear, so the network learns three *basis* flows around
the section, each with the body a streamline and no circulation of its own beyond what it is
built with:

    psi_x = y + psi'_x            (unit freestream along x, no circulation)
    psi_y = -x + psi'_y           (unit freestream along y, no circulation)
    psi_c = L(x, y) / (2 pi) + psi'_c   (unit circulation)

with psi' the network's three outputs. The flow at incidence alpha is

    psi = cos(alpha) psi_x + sin(alpha) psi_y + Gamma psi_c,

and Gamma is not learned: it is the value that makes the superposed flow satisfy the Kutta
condition, in closed form from the three basis flows' velocity jumps at the trailing edge
(:meth:`JoukowskiPINN.update_kutta`), exactly as a panel method fixes its circulation. None of
the basis flows sees a Kutta loss, so none can meet it with a local distortion at the edge.

Earlier formulations, kept here because the pilot found each failure: (1) a learnable Gamma
with a point vortex at mid-chord: the vortex sits about 0.05 chord from the surface of this
thin section and induces speeds near 1.75 there, so the network kept Gamma near zero and
carried the circulation in psi' (cancelling it in the sparse far field); (2) the same with a
vortex sheet and a zero-flux condition: Gamma rose but stalled near 0.4 of 0.55, because the
network met the Kutta loss with a non-harmonic patch at the edge, which the mean Laplace
residual barely charges. L is the mean of ln|z - s| over a uniform vortex sheet s on the camber
line from 0.1 to 0.9 chord (the bound vortex sheet of thin-airfoil theory, uniform strength):
ln r far away, like a point vortex, but without its near-surface speeds.

psi' is the network, and the question RES-01 asks of this case is how well it can be trained
when it sees the plane through ordinary coordinates. ``features`` chooses them:

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

import numpy as np
import torch
from torch import nn

from pinncfd.external.geometry import JoukowskiSection
from pinncfd.nn import MLP
from pinncfd.pinn_airfoil_flowfield.model import X0, compress
from pinncfd.pinn_airfoil_flowfield.problem import SURFACE_OFFSET

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


SHEET_X = (0.1, 0.9)  # chordwise extent of the bound vortex sheet [chord]
SHEET_SEGMENTS = 8


def _segment_log_integral(wx: torch.Tensor, wy: torch.Tensor, length: float) -> torch.Tensor:
    """Integral of ln|w - t| for t from 0 to ``length``, w = wx + i wy (real arithmetic).

    The antiderivative is -Re[(w - t) log(w - t)] + t, and Re(w log w) = wx ln|w| - wy arg w.
    The branch cuts of the two arg terms cancel everywhere off the segment itself.
    """

    def re_wlogw(a, b):
        return 0.5 * a * torch.log(a * a + b * b) - b * torch.atan2(b, a)

    return re_wlogw(wx, wy) - re_wlogw(wx - length, wy) - length


def sheet_log(xy: torch.Tensor, nodes: list[tuple[float, float]]) -> torch.Tensor:
    """Mean of ln|z - s| over a uniform vortex sheet on the polyline ``nodes`` (column).

    Gamma / (2 pi) times this is the stream function of circulation Gamma spread evenly along
    the sheet. Far away it is ln r + O(1/r), as for a point vortex.
    """
    x, y = xy[:, :1], xy[:, 1:2]
    total = sum(math.dist(p, q) for p, q in zip(nodes[:-1], nodes[1:], strict=True))
    out = 0.0
    for (x0, y0), (x1, y1) in zip(nodes[:-1], nodes[1:], strict=True):
        length = math.dist((x0, y0), (x1, y1))
        ex, ey = (x1 - x0) / length, (y1 - y0) / length
        dx, dy = x - x0, y - y0
        out = out + _segment_log_integral(dx * ex + dy * ey, -dx * ey + dy * ex, length)
    return out / total


# ---- the Kutta condition ----------------------------------------------------------------------
# The jump in downstream tangential velocity across the trailing edge, measured just above and
# below the surface at 8 stations from x = 0.85 to 0.985, fitted by least squares with a line in
# s = sqrt(1 - x), and evaluated at the edge (s = 0). The airfoil family's condition (the jump
# at x = 0.97 and 0.985 themselves) is biased on a *cusped* edge: a loaded cusp keeps a jump
# that shrinks only like sqrt(1 - x), so the exact flow leaves it at 1.3e-2 squared at 4
# degrees, while a flow with 10 % less circulation scores 1.4e-3. The extrapolated jump is 6e-4
# for the exact flow and 0.15 for 90 % of its circulation (``tests/test_joukowski_pinn.py``).
KUTTA_STATIONS = tuple(np.linspace(0.85, 0.985, 8))
KUTTA_S = np.sqrt(1 - np.array(KUTTA_STATIONS))
# least-squares weights of the fitted line's value at s = 0
KUTTA_WEIGHTS = tuple(np.linalg.pinv(np.vander(KUTTA_S, 2, increasing=True))[0])
GAMMA_EPS = 1e-3  # regularises Gamma = -A / K while K (about 2.7 when trained) is near 0


def kutta_points(section: JoukowskiSection, h: float = 1e-5) -> np.ndarray:
    """Upper then lower surface points at KUTTA_STATIONS, moved SURFACE_OFFSET outward, with
    their downstream unit tangents, as rows (x, y, tx, ty)."""
    xs = np.array(KUTTA_STATIONS)
    xu, yu, xl, yl = section.surfaces(xs)
    xu2, yu2, xl2, yl2 = section.surfaces(xs + h)
    rows = []
    for side, (x, y, x2, y2) in enumerate(((xu, yu, xu2, yu2), (xl, yl, xl2, yl2))):
        tx, ty = x2 - x, y2 - y
        norm = np.hypot(tx, ty)
        tx, ty = tx / norm, ty / norm
        sgn = 1.0 if side == 0 else -1.0
        nx, ny = -sgn * ty, sgn * tx
        rows.append(np.column_stack([x + SURFACE_OFFSET * nx, y + SURFACE_OFFSET * ny, tx, ty]))
    return np.vstack(rows)


def trailing_edge_jump(vt_upper, vt_lower):
    """Jump of the downstream tangential velocity at the Kutta stations (last axis), fitted with
    a line in s = sqrt(1 - x) and evaluated at the edge. Works on arrays and tensors."""
    d = vt_upper - vt_lower
    return sum(w * d[..., i] for i, w in enumerate(KUTTA_WEIGHTS))


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
        xs = np.linspace(*SHEET_X, SHEET_SEGMENTS + 1)
        _, yu, _, yl = self.section.surfaces(xs)
        self.sheet_nodes = [(float(x), float(y)) for x, y in zip(xs, (yu + yl) / 2, strict=True)]
        if features == "exact_map":  # rho in [0, ~2.5], cos, sin
            center, scale = [1.25, 0.0, 0.0], [1.25, 1.0, 1.0]
        else:
            center, scale = [0.0, 0.0], [1.0, 1.0]
        mlp = dict(dict(width=64, depth=5, activation="tanh"), **(mlp or {}))
        mlp.update(in_dim=len(center), out_dim=3, input_center=center, input_scale=scale)
        self.field = MLP(**mlp)
        self.psi_b = nn.Parameter(torch.zeros(3))  # body value of each basis flow
        xk = kutta_points(self.section)
        self.register_buffer("xy_kutta", torch.tensor(xk[:, :2], dtype=torch.float32))
        self.register_buffer("t_kutta", torch.tensor(xk[:, 2:], dtype=torch.float32))
        # the basis flows' extrapolated trailing-edge jumps (x, y, circulation)
        self.register_buffer("kutta_k", torch.tensor([0.0, 0.0, 1.0]))
        self.config = dict(
            alpha=self.alpha,
            features=features,
            mlp=self.field.config,
            mx=self.section.mx,
            my=self.section.my,
        )

    # ---- incidence, speed and circulation -----------------------------------------------------
    def incidence(self) -> torch.Tensor:
        return torch.tensor(self.alpha, dtype=self.kutta_k.dtype, device=self.kutta_k.device)

    def speed(self) -> torch.Tensor | float:
        return 1.0

    def gamma_for(self, a: torch.Tensor) -> torch.Tensor:
        """The circulation that satisfies the Kutta condition at incidence ``a``."""
        kx, ky, kc = self.kutta_k
        return -(torch.cos(a) * kx + torch.sin(a) * ky) * kc / (kc * kc + GAMMA_EPS**2)

    @property
    def gamma(self) -> torch.Tensor:
        """The circulation Gamma (clockwise positive, chord 1, unit freestream)."""
        return self.gamma_for(self.incidence())

    def coefficients(self) -> torch.Tensor:
        a = self.incidence()
        return self.speed() * torch.stack([torch.cos(a), torch.sin(a), self.gamma_for(a)])

    def update_kutta(self) -> None:
        """Recompute the basis flows' trailing-edge jumps (detached: the Kutta condition fixes
        Gamma, it does not train the network)."""
        xy = self.xy_kutta.detach().clone().requires_grad_(True)
        jumps = []
        for k in range(3):
            g = torch.autograd.grad(self.basis(xy)[:, k].sum(), xy)[0]
            vt = g[:, 1] * self.t_kutta[:, 0] - g[:, 0] * self.t_kutta[:, 1]
            vt = vt.view(2, len(KUTTA_STATIONS))
            jumps.append(trailing_edge_jump(vt[0], vt[1]).detach())
        with torch.no_grad():
            self.kutta_k.copy_(torch.stack(jumps))

    # ---- stream functions ---------------------------------------------------------------------
    def inputs(self, xy: torch.Tensor) -> torch.Tensor:
        if self.features == "plain":
            return torch.cat([xy[:, :1] - X0, xy[:, 1:2]], dim=1)
        if self.features == "compress":
            return compress(xy)
        return exact_map_features(xy, self.section)

    def perturbation(self, xy: torch.Tensor) -> torch.Tensor:
        """psi' of the three basis flows, (N, 3)."""
        return self.field(self.inputs(xy))

    def known_part(self, xy: torch.Tensor) -> torch.Tensor:
        """The basis flows' known parts (y, -x, L / (2 pi)), (N, 3)."""
        x, y = xy[:, :1], xy[:, 1:2]
        return torch.cat([y, -x, sheet_log(xy, self.sheet_nodes) / (2 * math.pi)], dim=1)

    def basis(self, xy: torch.Tensor) -> torch.Tensor:
        return self.known_part(xy) + self.perturbation(xy)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.basis(xy) @ self.coefficients()[:, None]

    def velocity(self, xy: torch.Tensor):
        """(u, v) = (d psi/dy, -d psi/dx); ``xy`` must require grad."""
        (g,) = torch.autograd.grad(
            self(xy), xy, grad_outputs=torch.ones(len(xy), 1, device=xy.device), create_graph=True
        )
        return g[:, 1:2], -g[:, 0:1]


def exact_basis(xy: torch.Tensor, section: JoukowskiSection) -> torch.Tensor:
    """The exact basis flows (psi_x, psi_y, psi_c) in torch, (N, 3), up to constants: the unit
    freestream along x and along y without circulation, and the unit circulation."""
    a2 = section.foil.radius**2
    w = circle_plane(xy, section)
    psi_x = (w + a2 / w).imag
    psi_y = (-1j * w + 1j * a2 / w).imag
    psi_c = section.chord / (2 * math.pi) * torch.log(w.abs())
    return torch.cat([psi_x, psi_y, psi_c], dim=1) / section.chord


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
