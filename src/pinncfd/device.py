"""Device selection: CUDA when available, overridable with ``PINNCFD_DEVICE``."""

from __future__ import annotations

import os

import torch


def get_device(prefer: str | None = None) -> torch.device:
    name = prefer or os.environ.get("PINNCFD_DEVICE")
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
