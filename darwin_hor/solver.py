from __future__ import annotations

import numpy as np

from darwin_hor.config import EvoConfig
from darwin_hor.geometry import Individual, Polar, max_camber, max_thickness


def evaluate_polar(airfoil, cfg: EvoConfig) -> Polar:
    alphas = np.asarray(cfg.alphas(), dtype=float)
    aero = airfoil.get_aero_from_neuralfoil(
        alpha=alphas,
        Re=cfg.re,
        mach=cfg.mach,
        model_size=cfg.model_size,
        include_360_deg_effects=False,
    )
    return Polar(
        alpha=alphas,
        cl=np.asarray(aero["CL"], dtype=float).reshape(-1),
        cd=np.asarray(aero["CD"], dtype=float).reshape(-1),
        cm=np.asarray(aero["CM"], dtype=float).reshape(-1),
        confidence=np.asarray(aero["analysis_confidence"], dtype=float).reshape(-1),
    )


def score_polar(polar: Polar, cfg: EvoConfig, thickness: float) -> tuple[float, float, float, float]:
    """Retourne (fitness, cl_cd_mean, cl_cd_design, confidence_mean)."""
    ratio = np.clip(polar.cl_cd, -400.0, 400.0)
    conf = np.clip(polar.confidence, 1e-6, 1.0)
    cl_cd_mean = float(np.average(ratio, weights=conf))
    design = polar.at_alpha(cfg.design_alpha())
    cl_cd_design = float(design["cl_cd"])
    confidence_mean = float(np.mean(conf))

    if cfg.fitness_mode == "classique":
        raw = float(np.sum(np.abs(ratio)))
    elif cfg.fitness_mode == "design":
        raw = cl_cd_design
    else:
        raw = cl_cd_mean

    penalty = 1.0
    if thickness > cfg.thickness_max:
        over = (thickness - cfg.thickness_max) / max(cfg.thickness_max, 1e-6)
        penalty *= 1.0 / (1.0 + 12.0 * over)
    if thickness < cfg.thickness_min:
        under = (cfg.thickness_min - thickness) / max(cfg.thickness_min, 1e-6)
        penalty *= 1.0 / (1.0 + 8.0 * under)
    if confidence_mean < 0.45:
        penalty *= 0.35 + 0.65 * (confidence_mean / 0.45)

    return raw * penalty, cl_cd_mean, cl_cd_design, confidence_mean


def evaluate_individual(ind: Individual, cfg: EvoConfig) -> Individual:
    try:
        ind.thickness = max_thickness(ind.airfoil)
        ind.camber = max_camber(ind.airfoil)
        ind.polar = evaluate_polar(ind.airfoil, cfg)
        fit, mean, design, conf = score_polar(ind.polar, cfg, ind.thickness)
        ind.fitness = fit
        ind.cl_cd_mean = mean
        ind.cl_cd_design = design
        ind.confidence_mean = conf
    except Exception:
        ind.fitness = 0.0
        ind.cl_cd_mean = 0.0
        ind.cl_cd_design = 0.0
        ind.confidence_mean = 0.0
        ind.polar = None
    return ind
