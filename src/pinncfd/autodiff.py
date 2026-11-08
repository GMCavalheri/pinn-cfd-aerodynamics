"""Derivatives of network outputs with respect to their inputs.

PDE residuals need exact derivatives of the network output, which reverse-mode automatic
differentiation provides at the cost of one backward pass per first derivative. Every helper
keeps the graph (``create_graph=True``) so the residual itself can be differentiated with
respect to the weights during training.
"""

from __future__ import annotations

import torch


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d(sum y)/dx: for a column ``y`` (N, 1) of pointwise outputs this is the (N, d) gradient."""
    (g,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)
    return g


def partials(y: torch.Tensor, x: torch.Tensor) -> list[torch.Tensor]:
    """First partials of ``y`` (N, 1) as a list of (N, 1) columns, one per input dimension."""
    g = grad(y, x)
    return [g[:, i : i + 1] for i in range(x.shape[1])]


def laplacian(y: torch.Tensor, x: torch.Tensor, dims=(0, 1)) -> torch.Tensor:
    """Sum of the second derivatives of ``y`` over the input columns ``dims`` (N, 1)."""
    g = grad(y, x)
    out = torch.zeros_like(y)
    for i in dims:
        out = out + grad(g[:, i : i + 1], x)[:, i : i + 1]
    return out
