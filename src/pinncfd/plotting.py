"""Shared matplotlib styling and plot helpers for the validation figures.

The style is the one used across the AERO series (copied from AERO-01 so every figure reads
as one set).

Series colours follow a fixed categorical order (never cycled or re-assigned by rank), lines
are thin, and grids/axes are recessive, so every figure in the repo reads as one set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#0b0b0b"  # exact / published reference curves are drawn in ink, dashed

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_MUTED,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": TEXT,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "font.size": 10,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}

FIGURES = Path(__file__).resolve().parents[2] / "docs" / "figures"


def use_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig, name: str, directory: Path = FIGURES) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cp(ax, x, cp, color, label=None, **kw):
    """Plot Cp with the aerodynamic convention (negative up)."""
    ax.plot(x, cp, color=color, label=label, **kw)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_ylabel("Pressure coefficient Cp")


def plot_body(ax, nodes, color=TEXT_MUTED):
    ax.fill(nodes[:, 0], nodes[:, 1], color=GRID, zorder=3)
    ax.plot(nodes[:, 0], nodes[:, 1], color=color, lw=1.0, zorder=4)
    ax.set_aspect("equal")


def convergence_plot(ax, history, key="error", label=None, color=SERIES[0], adam_steps=None):
    """Error (log scale) against the global step, with the Adam/L-BFGS boundary marked."""
    steps = np.array([h["step"] for h in history])
    vals = np.array([h[key] for h in history])
    ax.semilogy(steps, vals, color=color, label=label)
    if adam_steps:
        ax.axvline(adam_steps, color=TEXT_MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Step (Adam, then L-BFGS iterations)")
