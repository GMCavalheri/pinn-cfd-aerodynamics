"""Seeding. CUDA kernels are not bit-reproducible across runs, so GPU results are reported as
mean and spread over several seeds rather than as single bit-exact numbers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - for any third-party code on the legacy global RNG
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
