"""Exact potential flow around a Joukowski airfoil, everywhere in the fluid.

AERO-01 gives the exact *surface* pressure of a Joukowski section; RES-01 case 2 needs the
whole field, so the PINN's velocity error can be measured off the body. The construction is
the classical one (Anderson, *Fundamentals of Aerodynamics*, Sec. 4.14; Katz & Plotkin,
Sec. 6.6; Milne-Thomson, *Theoretical Aerodynamics*, Ch. 6):

* the map z = zeta + b^2 / zeta sends the circle |zeta - zeta0| = a, which passes through
  zeta = b, onto the airfoil, with the cusped trailing edge at z = 2b;
* in the circle plane, the flow at incidence alpha with clockwise circulation Gamma has the
  complex potential (zeta' = zeta - zeta0)

      F = zeta' e^(-i alpha) + a^2 e^(i alpha) / zeta' + (i Gamma / 2 pi) ln zeta';

* the Kutta condition (a stagnation point at zeta = b, where dz/dzeta = 0) fixes
  Gamma = 4 pi a sin(alpha + beta);
* the velocity in the airfoil plane is u - i v = (dF/dzeta) / (dz/dzeta), and the stream
  function is psi = Im F, single-valued because Im(i c ln zeta') = c ln|zeta'|.

Everything is returned in the normalised frame of :class:`JoukowskiSection` (chord 1, leading
edge at x = 0, freestream speed 1): velocities are unchanged by the scaling, the stream
function and the circulation are divided by the chord, and cl = 2 Gamma.
"""

from __future__ import annotations

import numpy as np

from pinncfd.external.geometry import JoukowskiSection


class JoukowskiFlow:
    def __init__(self, section: JoukowskiSection | None = None):
        self.section = section or JoukowskiSection()
        foil = self.section.foil
        self.foil = foil
        self.a, self.b, self.zeta0 = foil.radius, foil.b, foil.zeta0
        self.chord, self.x_le = self.section.chord, self.section.x_le

    # ---- maps ------------------------------------------------------------------------------
    def to_circle(self, x, y) -> np.ndarray:
        """Circle-plane zeta of normalised points (x, y) outside the section.

        Inverts z = zeta + b^2/zeta, zeta = (z +- sqrt(z^2 - 4 b^2)) / 2, keeping the root
        outside the circle (the same choice as AERO-01's ``surface_cp``).
        """
        z = (np.asarray(x) + 1j * np.asarray(y)) * self.chord + self.x_le
        disc = np.sqrt(z * z - 4 * self.b**2)
        r1, r2 = (z + disc) / 2, (z - disc) / 2
        return np.where(np.abs(r1 - self.zeta0) >= np.abs(r2 - self.zeta0), r1, r2)

    def from_circle(self, zeta) -> tuple[np.ndarray, np.ndarray]:
        """Normalised (x, y) of circle-plane points."""
        z = (zeta + self.b**2 / zeta - self.x_le) / self.chord
        return z.real, z.imag

    def body_points(self, n: int, skip_te: float = 0.0, phase: float = 0.0):
        """(x, y, theta) of n points on the body, equispaced in the circle angle theta measured
        from the trailing-edge point and shifted by ``phase`` spacings (0.5 interleaves a second
        set between the first); ``skip_te`` drops |theta| < skip_te (the 0/0 at the cusp)."""
        th = (np.arange(n) + phase) * 2 * np.pi / n
        th = th[np.minimum(th, 2 * np.pi - th) >= skip_te]
        te_angle = np.angle(self.b - self.zeta0)
        zeta = self.zeta0 + self.a * np.exp(1j * (te_angle + th))
        x, y = self.from_circle(zeta)
        return x, y, th

    # ---- the flow --------------------------------------------------------------------------
    def circulation(self, alpha: float) -> float:
        """Clockwise circulation in the normalised frame (chord 1, V = 1)."""
        return float(self.foil.circulation(alpha) / self.chord)

    def cl(self, alpha: float) -> float:
        return float(self.foil.cl(alpha))

    def body_value(self, alpha: float) -> float:
        """psi on the body: Im F = (Gamma / 2 pi) ln a on |zeta'| = a, over the chord."""
        return float(self.foil.circulation(alpha) / (2 * np.pi) * np.log(self.a) / self.chord)

    def stream_function(self, x, y, alpha: float) -> np.ndarray:
        zp = self.to_circle(x, y) - self.zeta0
        g = self.foil.circulation(alpha)
        f = zp * np.exp(-1j * alpha) + self.a**2 * np.exp(1j * alpha) / zp
        return (f.imag + g / (2 * np.pi) * np.log(np.abs(zp))) / self.chord

    def velocity(self, x, y, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        zeta = self.to_circle(x, y)
        zp = zeta - self.zeta0
        g = self.foil.circulation(alpha)
        dfdzeta = (
            np.exp(-1j * alpha) - self.a**2 * np.exp(1j * alpha) / zp**2 + 1j * g / (2 * np.pi * zp)
        )
        w = dfdzeta / (1 - self.b**2 / zeta**2)
        return w.real, -w.imag

    def cp(self, x, y, alpha: float) -> np.ndarray:
        u, v = self.velocity(x, y, alpha)
        return 1.0 - u**2 - v**2
