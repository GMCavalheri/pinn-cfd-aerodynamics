"""Fully connected networks for PINNs.

A PINN is an ordinary multilayer perceptron whose inputs are coordinates (and, for a family of
problems, parameters such as the airfoil shape and angle of attack). What makes it physics-
informed is the loss, not the architecture, so one configurable MLP covers every phase:

* smooth activations (``tanh``, ``sin``, ``silu``): the loss contains second derivatives of the
  output with respect to the input, so ReLU (whose second derivative is zero almost everywhere)
  cannot be used;
* an optional random Fourier-feature input layer (Tancik et al. 2020), which counters the
  spectral bias of MLPs toward low frequencies;
* a fixed affine input map, so the network always sees inputs in roughly [-1, 1].

Weights use Glorot (Xavier) normal initialisation, the usual choice for tanh networks
(Glorot & Bengio 2010); ``sin`` networks use the SIREN scheme (Sitzmann et al. 2020).
"""

from __future__ import annotations

import math

import torch
from torch import nn

ACTIVATIONS = ("tanh", "sin", "silu")


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def make_activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "sin":
        return Sine()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unknown activation {name!r}; choose from {ACTIVATIONS}")


class FourierFeatures(nn.Module):
    """x -> [sin(2 pi x B), cos(2 pi x B)] with a fixed Gaussian matrix B ~ N(0, scale^2)."""

    def __init__(self, in_dim: int, n_features: int, scale: float, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", scale * torch.randn(in_dim, n_features, generator=g))

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, x):
        z = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)


class MLP(nn.Module):
    """``depth`` hidden layers of ``width`` units between an input map and a linear output.

    ``input_center`` / ``input_scale`` define the affine map ``(x - center) / scale`` applied
    before anything else (per input column).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        width: int = 50,
        depth: int = 4,
        activation: str = "tanh",
        fourier_features: int = 0,
        fourier_scale: float = 1.0,
        input_center=None,
        input_scale=None,
        seed: int = 0,
    ):
        super().__init__()
        self.config = dict(
            in_dim=in_dim,
            out_dim=out_dim,
            width=width,
            depth=depth,
            activation=activation,
            fourier_features=fourier_features,
            fourier_scale=fourier_scale,
            input_center=None if input_center is None else list(map(float, input_center)),
            input_scale=None if input_scale is None else list(map(float, input_scale)),
            seed=seed,
        )
        center = torch.zeros(in_dim) if input_center is None else torch.tensor(input_center)
        scale = torch.ones(in_dim) if input_scale is None else torch.tensor(input_scale)
        self.register_buffer("center", center.float())
        self.register_buffer("scale", scale.float())

        g = torch.Generator().manual_seed(seed)
        self.features = (
            FourierFeatures(in_dim, fourier_features, fourier_scale, seed=seed)
            if fourier_features
            else None
        )
        first = self.features.out_dim if self.features else in_dim
        dims = [first] + [width] * depth + [out_dim]
        pairs = zip(dims[:-1], dims[1:], strict=True)
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in pairs)
        self.act = make_activation(activation)
        for lin in self.linears:
            if activation == "sin":
                # SIREN's hidden-layer rule U(-sqrt(6/n), sqrt(6/n)) with w0 = 1 everywhere:
                # the fields here are smooth, so SIREN's high-frequency first layer
                # (w0 = 30) is not wanted.
                bound = math.sqrt(6.0 / lin.in_features)
                with torch.no_grad():
                    lin.weight.uniform_(-bound, bound, generator=g)
            else:
                std = math.sqrt(2.0 / (lin.in_features + lin.out_features))
                with torch.no_grad():
                    lin.weight.normal_(0.0, std, generator=g)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        h = (x - self.center) / self.scale
        if self.features is not None:
            h = self.features(h)
        for lin in self.linears[:-1]:
            h = self.act(lin(h))
        return self.linears[-1](h)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
