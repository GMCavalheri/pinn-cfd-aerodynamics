"""RES-01 case 2: one Joukowski section, exact potential flow and a PINN on plain coordinates."""

from pinncfd.pinn_joukowski.exact import JoukowskiFlow
from pinncfd.pinn_joukowski.model import JoukowskiPINN
from pinncfd.pinn_joukowski.problem import JoukowskiProblem

__all__ = ["JoukowskiFlow", "JoukowskiPINN", "JoukowskiProblem"]
