"""A PINN for the stream function around any section of the NACA family at any incidence.

The stream function is split into a known part and a learned part,

    psi(x, y; q) = y cos(alpha) - x sin(alpha) + Gamma(q) / (2 pi) ln(|w| / R) + psi'(x, y; q),

where q = (m, p, t, alpha) and w(x, y; q) is the inverse Joukowski map of a circular-arc slit
inside the section (``joukowski_map``), whose image is a circle of radius R. ln|w| is harmonic
in the fluid and behaves like ln r far away, so the second term is a point vortex of clockwise
circulation Gamma "centred in the section". Gamma and the body value psi_b come from a small
second network of q alone. psi' is the main network: the disturbance, which decays like a
doublet and is O(thickness) everywhere.

Gamma is a convenient split, not a measured quantity: psi' can carry part of the circulation
in the layer next to the body, so the lift is always measured from the predicted field
(circulation around a contour, or surface pressure), never read off the head.

The main network does not see (x, y) but the log-polar coordinates (ln|w|/R, w/|w|) of the
circle plane. There the section is a near-circle and the flow is smooth; the sharp variations
at the leading and trailing edges are carried by the map's derivative, and a flow violating the
Kutta condition becomes singular at the trailing edge, where the Kutta loss sees it. With plain
coordinates (``features="compress"``: (x - x0, y) ln(1 + r/a)/r, a = 0.5, kept for comparison)
the network cannot resolve the leading edge and trains to a slowed-down, wrong flow.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pinncfd.nn import MLP
from pinncfd.pinn_airfoil_flowfield.family import HIGH, LOW

X0 = 0.5  # domain centre [chord]
R_FAR = 8.0  # outer boundary radius about (X0, 0) [chord]
COMPRESS = 0.5


def camber_at_midchord(m, p):
    """y_c(0.5) of a NACA 4-digit section with p <= 0.5 (Abbott & von Doenhoff Sec. 6.4)."""
    return m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * X0 - X0**2)


ARC_SAG = 0.9  # arc slit height as a fraction of the maximum camber


def joukowski_map(xy: torch.Tensor, q: torch.Tensor):
    """Inverse Joukowski map of the exterior of a circular-arc slit inside the section.

    The map z = zeta + b^2 / zeta sends a circle through zeta = +-b centred at i h onto a
    circular arc of height 2h between z = -2b and 2b (Milne-Thomson, *Theoretical
    Aerodynamics*, Ch. 6). Inverting it, zeta = (z +- sqrt(z^2 - 4 b^2)) / 2, with the root
    chosen outside the circle, maps the exterior of the arc conformally onto the exterior of
    the circle. The arc runs from x = 0.55 t^2 (half the leading-edge radius, inside the
    nose) to the trailing edge x = 1 with a height 0.9 m, which keeps it inside every section
    of the family, so the map is smooth everywhere in the fluid.

    Returns (rho, w) with w = zeta - i h (complex) and rho = ln(|w| / R) >= 0, R the circle
    radius. Both edges of the section map to smooth regions of the circle plane; the square-
    root behaviour of the flow at the leading and trailing edges is carried by d(zeta)/dz.
    """
    m, t = q[:, 0:1], q[:, 2:3]
    eps = 0.55 * t**2
    c = (1 + eps) / 2
    b = (1 - eps) / 4
    h = ARC_SAG * m / 2
    z = torch.complex(xy[:, 0:1] - c, xy[:, 1:2])
    s = torch.sqrt(z * z - (4 * b * b).to(z.dtype))
    zp, zm = (z + s) / 2, (z - s) / 2
    ih = torch.complex(torch.zeros_like(h), h)
    zeta = torch.where((zp - ih).abs() >= (zm - ih).abs(), zp, zm)
    w = zeta - ih
    rho = 0.5 * torch.log(w.abs() ** 2 / (b * b + h * h))
    return rho, w


def joukowski_features(xy: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(rho, cos phi, sin phi) of w = zeta - i h: log-polar coordinates of the circle plane."""
    rho, w = joukowski_map(xy, q)
    unit = w / w.abs()
    return torch.cat([rho, unit.real, unit.imag], dim=1)


def compress(xy: torch.Tensor) -> torch.Tensor:
    dx = xy[:, :1] - X0
    dy = xy[:, 1:2]
    r = torch.sqrt(dx**2 + dy**2 + 1e-12)
    g = torch.log1p(r / COMPRESS) / r
    return torch.cat([dx * g, dy * g], dim=1)


class AirfoilPINN(nn.Module):
    def __init__(
        self,
        mlp: dict | None = None,
        head: dict | None = None,
        features: str = "joukowski",
    ):
        super().__init__()
        if features not in ("joukowski", "compress"):
            raise ValueError("features must be 'joukowski' or 'compress'")
        self.features = features
        qc, qs = list((LOW + HIGH) / 2), list((HIGH - LOW) / 2)
        if features == "joukowski":  # rho in [0, ~3.5], cos, sin
            center, scale = [1.5, 0.0, 0.0] + qc, [1.5, 1.0, 1.0] + qs
        else:
            center, scale = [0.0, 0.0] + qc, [1.0, 1.0] + qs
        mlp = dict(dict(width=64, depth=5, activation="tanh"), **(mlp or {}))
        mlp.update(in_dim=len(center), out_dim=1, input_center=center, input_scale=scale)
        head = dict(dict(width=32, depth=2, activation="tanh"), **(head or {}))
        head.update(in_dim=4, out_dim=2, input_center=qc, input_scale=qs)
        self.field = MLP(**mlp)
        self.head = MLP(**head)
        self.config = dict(
            mlp=self.field.config,
            head=self.head.config,
            features=features,
        )

    def case_scalars(self, q: torch.Tensor):
        """(Gamma, psi_b) for parameter rows q = (m, p, t, alpha)."""
        out = self.head(q)
        return out[:, :1], out[:, 1:2]

    def perturbation(self, xy, q):
        f = joukowski_features(xy, q) if self.features == "joukowski" else compress(xy)
        return self.field(torch.cat([f, q], dim=1))

    def known_part(self, xy, q, gamma):
        x, y = xy[:, :1], xy[:, 1:2]
        alpha = q[:, 3:4]
        if self.features == "joukowski":
            log_r = joukowski_map(xy, q)[0]  # ln|w| - ln R: harmonic, ~ ln r far away
        else:
            yv = camber_at_midchord(q[:, :1], q[:, 1:2])
            log_r = 0.5 * torch.log((x - X0) ** 2 + (y - yv) ** 2)
        return y * torch.cos(alpha) - x * torch.sin(alpha) + gamma / (2 * math.pi) * log_r

    def forward(self, xy, q):
        gamma, _ = self.case_scalars(q)
        return self.known_part(xy, q, gamma) + self.perturbation(xy, q)

    def velocity(self, xy, q):
        """(u, v) = (d psi/dy, -d psi/dx); ``xy`` must require grad."""
        (g,) = torch.autograd.grad(
            self(xy, q),
            xy,
            grad_outputs=torch.ones(len(xy), 1, device=xy.device),
            create_graph=True,
        )
        return g[:, 1:2], -g[:, 0:1]
