from __future__ import annotations

from dataclasses import dataclass, field


FITNESS_MODES = ("moyenne", "design", "classique")
MODEL_SIZES = (
    "xxsmall",
    "xsmall",
    "small",
    "medium",
    "large",
    "xlarge",
    "xxlarge",
    "xxxlarge",
)


@dataclass
class EvoConfig:
    """Paramètres d'une campagne Darwin.

    `thickness_max` / `thickness_min` sont des fractions de corde
    (0.22 = 22 %). Dans la V2 Scilab le dialogue demandait 22, mais le
    filtre comparait max(y) à 11 : il ne s'activait presque jamais.
    Ici 22 % veut vraiment dire 22 % d'épaisseur.
    """

    n_generations: int = 20
    pop_size: int = 12
    re: float = 3.0e5
    mach: float = 0.2
    alpha_min: float = 0.0
    alpha_max: float = 11.0
    alpha_step: float = 1.0
    alpha_design: float | None = None
    thickness_max: float = 0.22
    thickness_min: float = 0.08
    model_size: str = "large"
    fitness_mode: str = "moyenne"
    weak_survival: float = 0.05
    elite_death: float = 0.01
    n_elites: int = 2
    naca_injection: int = 4
    seed: int = 42
    chord_export_mm: float = 100.0
    n_points: int = 121

    def __post_init__(self) -> None:
        # Les curseurs UI sont en % (8 = 8 %). Si on reçoit 8 au lieu de 0.08, on convertit.
        if self.thickness_max > 1.0:
            self.thickness_max /= 100.0
        if self.thickness_min > 1.0:
            self.thickness_min /= 100.0
        if self.thickness_min >= self.thickness_max:
            self.thickness_min = max(0.04, self.thickness_max - 0.04)

    def design_alpha(self) -> float:
        if self.alpha_design is not None:
            return float(self.alpha_design)
        return 0.5 * (self.alpha_min + self.alpha_max)

    def alphas(self):
        import numpy as np

        start = self.alpha_min
        stop = self.alpha_max + 0.5 * self.alpha_step
        return np.arange(start, stop, self.alpha_step)


@dataclass
class BladeConfig:
    """Pale 3D par BEM / Schmitz, à partir du profil champion."""

    radius_m: float = 0.40
    hub_radius_m: float = 0.05
    n_blades: int = 3
    tsr: float = 6.0
    n_stations: int = 12
    cl_fallback: float = 0.8
    chord_max_over_r: float = 0.35
    pitch_axis_xc: float = 0.25
    n_airfoil_points: int = 80


@dataclass
class CampaignMeta:
    title: str = "Darwin HOR 3"
    author: str = "Stéphane Roddier (IUT GIM Saint-Denis)"
    notes: str = ""
    extra: dict = field(default_factory=dict)
