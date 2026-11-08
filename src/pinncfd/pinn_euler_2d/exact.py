"""Exact solutions of the steady 2D Euler equations used to verify the PINN (gamma = 1.4).

**Ringleb flow** (Ringleb 1940; Chiocchia 1985, AGARD AR-211) is a smooth, isentropic,
irrotational flow obtained by the hodograph method. In terms of the speed q and the streamline
label k (each streamline is k = const, with q <= k), and the sound speed c, with
c^2 = 1 - (gamma - 1)/2 q^2 (stagnation sound speed 1):

    rho = c^(2/(gamma-1)),  p = c^(2 gamma/(gamma-1)) / gamma,
    J = 1/c + 1/(3c^3) + 1/(5c^5) - (1/2) ln((1 + c)/(1 - c)),
    x = (1/(2 rho)) (1/q^2 - 2/k^2) + J/2,   y = +-(1/(k rho q)) sqrt(1 - q^2/k^2),
    u = sign(y) q cos(theta), v = q sin(theta), sin(theta) = q/k.

The flow turns around the end of a plate; it is supersonic where q > sqrt(2/(gamma+1)) = 0.913,
so with streamlines up to k = 1.2 the domain contains a smooth supersonic pocket.

**Oblique shock on a wedge** (NACA Report 1135, Ames Research Staff 1953; Anderson, *Modern
Compressible Flow*, Ch. 4): a uniform supersonic stream of Mach M meets a wedge of half-angle
theta at the origin; an attached straight shock at angle beta (weak root of the theta-beta-M
relation) turns it parallel to the wall, with the Rankine-Hugoniot jumps across it.

Non-dimensionalisation for the wedge: freestream rho = 1, sound speed 1 (p = 1/gamma), u = M.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

GAMMA = 1.4

# ---------------------------------------------------------------- Ringleb
K_MIN, K_MAX, Q0 = 0.7, 1.2, 0.5


def ringleb(q, k, branch):
    """(x, y, rho, u, v, p) for speed q, streamline k and branch +1 (y > 0) or -1."""
    g = GAMMA
    q, k, branch = np.broadcast_arrays(np.asarray(q, float), np.asarray(k, float), branch)
    c = np.sqrt(1 - (g - 1) / 2 * q**2)
    rho = c ** (2 / (g - 1))
    p = c ** (2 * g / (g - 1)) / g
    J = 1 / c + 1 / (3 * c**3) + 1 / (5 * c**5) - 0.5 * np.log((1 + c) / (1 - c))
    x = 1 / (2 * rho) * (1 / q**2 - 2 / k**2) + J / 2
    s = np.clip(q / k, 0.0, 1.0)
    y = branch / (k * rho * q) * np.sqrt(1 - s**2)
    u = branch * q * np.sqrt(1 - s**2)
    v = q * s
    return x, y, rho, u, v, p


def ringleb_samples(n: int, rng: np.random.Generator):
    """Interior points: (q, k, branch) drawn so the points are roughly area-uniform in (x, y).

    Uniform (q, k) draws are thinned by the Jacobian |d(x, y)/d(q, k)| (rejection sampling)."""
    out, need = [], n
    h = 1e-6
    while need > 0:
        m = 4 * need
        k = K_MIN + (K_MAX - K_MIN) * rng.random(m)
        q = Q0 + (k - Q0) * rng.random(m) * (1 - 1e-6)
        b = np.where(rng.random(m) < 0.5, 1.0, -1.0)
        x, y, *_ = ringleb(q, k, b)
        xq, yq, *_ = ringleb(q + h, k, b)
        xk, yk, *_ = ringleb(q, k + h, b)
        jac = np.abs((xq - x) * (yk - y) - (xk - x) * (yq - y)) / h**2 * (k - Q0)
        keep = rng.random(m) < jac / np.percentile(jac, 99)
        out.append(np.column_stack([q[keep], k[keep], b[keep]])[:need])
        need -= out[-1].shape[0]
    return np.vstack(out)


def ringleb_boundary(n: int, rng: np.random.Generator):
    """Boundary (q, k, branch): the two walls k = K_MIN, K_MAX and the inflow/outflow q = Q0."""
    n3 = n // 3
    rows = []
    for k0 in (K_MIN, K_MAX):
        q = Q0 + (k0 - Q0) * rng.random(n3)
        b = np.where(rng.random(n3) < 0.5, 1.0, -1.0)
        rows.append(np.column_stack([q, np.full(n3, k0), b]))
    m = n - 2 * n3
    k = K_MIN + (K_MAX - K_MIN) * rng.random(m)
    b = np.where(rng.random(m) < 0.5, 1.0, -1.0)
    rows.append(np.column_stack([np.full(m, Q0), k, b]))
    return np.vstack(rows)


# ---------------------------------------------------------------- oblique shock
def theta_beta_m(beta, mach, gamma=GAMMA):
    """Flow deflection theta for shock angle beta (Anderson, Modern Compressible Flow, Ch. 4)."""
    m2 = mach**2
    return np.arctan(
        2 / np.tan(beta) * (m2 * np.sin(beta) ** 2 - 1) / (m2 * (gamma + np.cos(2 * beta)) + 2)
    )


def weak_shock_angle(theta, mach, gamma=GAMMA):
    """Weak-shock root beta of the theta-beta-M relation."""
    mu = np.arcsin(1 / mach)
    # theta(beta) rises from 0 at the Mach angle to its maximum; search below the maximum
    betas = np.linspace(mu + 1e-9, np.pi / 2, 20001)
    i_max = int(np.argmax(theta_beta_m(betas, mach, gamma)))
    return brentq(lambda b: theta_beta_m(b, mach, gamma) - theta, mu + 1e-9, betas[i_max])


def oblique_shock(theta, mach, gamma=GAMMA):
    """Shock angle and the post-shock state, freestream rho = 1, c = 1, p = 1/gamma."""
    beta = weak_shock_angle(theta, mach, gamma)
    mn1 = mach * np.sin(beta)
    rho_ratio = (gamma + 1) * mn1**2 / ((gamma - 1) * mn1**2 + 2)
    p_ratio = 1 + 2 * gamma / (gamma + 1) * (mn1**2 - 1)
    mn2 = np.sqrt((1 + (gamma - 1) / 2 * mn1**2) / (gamma * mn1**2 - (gamma - 1) / 2))
    m2 = mn2 / np.sin(beta - theta)
    rho2, p2 = rho_ratio, p_ratio / gamma
    c2 = np.sqrt(gamma * p2 / rho2)
    v2 = m2 * c2
    return {
        "beta": beta,
        "rho_ratio": rho_ratio,
        "p_ratio": p_ratio,
        "mach2": m2,
        "state1": np.array([1.0, mach, 0.0, 1 / gamma]),
        "state2": np.array([rho2, v2 * np.cos(theta), v2 * np.sin(theta), p2]),
    }


def wedge_field(x, y, theta, mach):
    """Exact primitive state (rho, u, v, p) at points above the wedge y >= x tan(theta)."""
    s = oblique_shock(theta, mach)
    behind = np.asarray(y) < np.asarray(x) * np.tan(s["beta"])
    return np.where(behind[..., None], s["state2"], s["state1"])
