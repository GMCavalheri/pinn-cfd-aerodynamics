"""The airfoil PINN used as a surrogate: many lift evaluations in one batched call.

A panel method solves one (shape, incidence) at a time: build the contour, assemble an N x N
influence matrix, solve. A trained PINN answers any number of cases with one forward and one
backward pass over all their surface points at once. These helpers turn a model into a
``cl(params)`` function in the two ways Lesson 2 validated:

* ``cl_pressure`` — Cp at the control points of an ``n``-panel contour of each section,
  integrated like the panel method's own pressure lift;
* ``cl_circulation`` — Kutta-Joukowski from the circulation on a circle (cheaper: no contour).

Both return float64 NumPy arrays and run on whatever device the model is on. ``solve_alpha``
inverts the first for the incidence: the network is differentiable in its inputs, so
d cl / d alpha comes from autograd and Newton's method runs on every section at once.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from pinncfd.external.geometry import naca
from pinncfd.pinn_airfoil_flowfield.problem import lift_from_circulation, predict_velocity


def surface_panels(params: np.ndarray, n: int):
    """Control points, outward normals and lengths of n-panel contours, stacked per case."""
    xy, nrm, ds = [], [], []
    for m, p, t, _ in params:
        nodes = naca(m, p, t).coordinates(n)
        d = np.diff(nodes, axis=0)
        length = np.hypot(d[:, 0], d[:, 1])
        xy.append(0.5 * (nodes[:-1] + nodes[1:]))
        nrm.append(np.column_stack([-d[:, 1], d[:, 0]]) / length[:, None])
        ds.append(length)
    return np.vstack(xy), np.vstack(nrm), np.concatenate(ds)


def _pressure_lift(cp, nrm, ds, alpha, lib):
    """cl from Cp at panel control points (cases, n): -sum(Cp n ds), rotated to the wind."""
    fx = -(cp * nrm[..., 0] * ds).sum(-1)
    fy = -(cp * nrm[..., 1] * ds).sum(-1)
    return fy * lib.cos(alpha) - fx * lib.sin(alpha)


def cl_pressure(model, params: np.ndarray, device, n: int = 120, chunk: int = 65536):
    """Lift from the predicted surface pressure for rows (m, p, t, alpha)."""
    params = np.asarray(params, dtype=float)
    xy, nrm, ds = surface_panels(params, n)
    u, v = predict_velocity(model, xy, np.repeat(params, n, axis=0), device, chunk=chunk)
    cp = (1.0 - u**2 - v**2).reshape(len(params), n)
    return _pressure_lift(cp, nrm.reshape(-1, n, 2), ds.reshape(-1, n), params[:, 3], np)


def solve_alpha(
    model, params: np.ndarray, target: float, device, n: int = 120, iters: int = 20, tol=1e-5
):
    """Incidence giving ``cl_pressure = target`` for each row (m, p, t, alpha0), by Newton.

    All rows iterate together; d cl / d alpha is the derivative of the network's pressure lift
    with respect to its alpha input, by autograd.

    Returns (alpha, cl at alpha, network evaluations)."""
    params = np.asarray(params, dtype=float)
    xy, nrm, ds = surface_panels(params, n)

    def t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    xy, nrm, ds = t(xy), t(nrm).reshape(-1, n, 2), t(ds).reshape(-1, n)
    shape = t(params[:, :3])
    alpha = t(params[:, 3])
    n_iter = 0
    while True:
        n_iter += 1
        a = alpha.clone().requires_grad_(True)
        q = torch.cat([shape, a[:, None]], dim=1).repeat_interleave(n, dim=0)
        u, v = model.velocity(xy.clone().requires_grad_(True), q)
        cp = (1.0 - u**2 - v**2).reshape(-1, n)
        cl = _pressure_lift(cp, nrm, ds, a, torch)
        residual = (cl - target).detach()
        if residual.abs().max() < tol or n_iter == iters:
            break  # alpha and cl are consistent: cl was evaluated at alpha
        (dcl,) = torch.autograd.grad(cl.sum(), a)  # rows are independent: the diagonal
        alpha = alpha - residual / dcl
    return alpha.cpu().double().numpy(), cl.detach().cpu().double().numpy(), n_iter


def cl_circulation(model, params: np.ndarray, device, n: int = 256):
    return lift_from_circulation(model, np.asarray(params, dtype=float), device, n)


def timed(fn, *args, repeats: int = 3, device=None, **kw):
    """(result, median wall time [s]) of ``fn(*args, **kw)`` after one warm-up call."""
    fn(*args, **kw)
    times = []
    for _ in range(repeats):
        if device is not None and torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kw)
        if device is not None and torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def break_even(train_s: float, per_case_panel_s: float, per_case_pinn_s: float) -> float:
    """Number of evaluations after which training + PINN inference beats the panel method.

    Infinite when the PINN is not faster per case."""
    gain = per_case_panel_s - per_case_pinn_s
    return float("inf") if gain <= 0 else train_s / gain
