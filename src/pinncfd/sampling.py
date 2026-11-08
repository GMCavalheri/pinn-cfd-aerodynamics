"""Collocation points: where the PDE residual and boundary conditions are enforced.

A PINN never builds a mesh; it evaluates the residual at scattered points that are redrawn
during training. Two things matter:

* **coverage** of the whole domain, so the residual is small everywhere, and
* **density** where the solution varies fastest — next to a body, the velocity changes on a
  length scale of the body radius or the leading-edge radius, while the far field is almost
  uniform.

The samplers below mix an area-uniform population with a population clustered against the
body. All return float64 NumPy arrays; the training code converts them.
"""

from __future__ import annotations

import numpy as np


def annulus(
    n: int, r_in: float, r_out: float, rng: np.random.Generator, near_fraction: float = 0.5
) -> np.ndarray:
    """``n`` points in r_in <= r <= r_out, part uniform in area, part clustered near r_in.

    The clustered part uses r = r_in + (r_out - r_in) u^3 with u uniform, which puts half of
    those points within an eighth of the gap from the body.
    """
    n_near = int(round(near_fraction * n))
    n_uni = n - n_near
    r_uni = np.sqrt(r_in**2 + rng.random(n_uni) * (r_out**2 - r_in**2))
    r_near = r_in + (r_out - r_in) * rng.random(n_near) ** 3
    r = np.concatenate([r_uni, r_near])
    th = rng.random(n) * 2 * np.pi
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def circle(n: int, radius: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """``n`` points on a circle: equispaced, or uniformly random if ``rng`` is given."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) if rng is None else 2 * np.pi * rng.random(n)
    return np.column_stack([radius * np.cos(th), radius * np.sin(th)])


def polar_grid(n_r: int, n_theta: int, r_in: float, r_out: float) -> np.ndarray:
    """Evaluation grid, equispaced in theta and geometrically stretched in r (dense at r_in).

    Used for error metrics: the stretching mirrors where the flow varies, so a relative L2 norm
    on it weighs the near field as much as the far field.
    """
    r = r_in * (r_out / r_in) ** np.linspace(0, 1, n_r)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rr, tt = np.meshgrid(r, th, indexing="ij")
    return np.column_stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()])
