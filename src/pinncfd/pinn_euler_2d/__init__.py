"""Phase 5 (exploratory): PINNs for the steady 2D Euler equations."""

from pinncfd.pinn_euler_2d.euler import EulerPINN, RinglebProblem, WedgeProblem, residual

__all__ = ["EulerPINN", "RinglebProblem", "WedgeProblem", "residual"]
