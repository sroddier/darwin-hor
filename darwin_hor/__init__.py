"""Darwin HOR 3 — évolution de profils de pale d'éolienne."""

from darwin_hor.config import BladeConfig, EvoConfig
from darwin_hor.evolution import EvolutionResult, run_evolution
from darwin_hor.geometry import Individual, load_dat, naca4, random_naca4

__version__ = "3.0.0"

__all__ = [
    "BladeConfig",
    "EvoConfig",
    "EvolutionResult",
    "Individual",
    "load_dat",
    "naca4",
    "random_naca4",
    "run_evolution",
    "__version__",
]
