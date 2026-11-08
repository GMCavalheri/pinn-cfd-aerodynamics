"""Phase 2: a PINN for the flow field around a parameterized NACA 4-digit family."""

from pinncfd.pinn_airfoil_flowfield.model import AirfoilPINN
from pinncfd.pinn_airfoil_flowfield.problem import AirfoilProblem, build_reference

__all__ = ["AirfoilPINN", "AirfoilProblem", "build_reference"]
