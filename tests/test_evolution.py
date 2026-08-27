import numpy as np

from darwin_hor.config import EvoConfig
from darwin_hor.evolution import run_evolution
from darwin_hor.geometry import interpolate_airfoils, naca4
from darwin_hor.solver import evaluate_individual, evaluate_polar


def test_interpolation_is_not_truncated_to_zero():
    a = naca4("4412", n_points=81)
    b = naca4("0012", n_points=81)
    mid = interpolate_airfoils(a, b, 0.5, "blend")
    ya = np.asarray(a.coordinates)[:, 1].max()
    yb = np.asarray(b.coordinates)[:, 1].max()
    ym = np.asarray(mid.coordinates)[:, 1].max()
    # A 50% blend must sit between the two camber/thickness envelopes, not copy A (the V2 bug).
    lo, hi = sorted((ya, yb))
    assert lo - 0.01 <= ym <= hi + 0.01
    assert abs(ym - ya) > 0.002


def test_neuralfoil_polar_naca4412():
    af = naca4("4412", n_points=81)
    cfg = EvoConfig(alpha_min=0, alpha_max=6, alpha_step=2, model_size="medium", re=3e5)
    polar = evaluate_polar(af, cfg)
    assert polar.cl.shape == polar.alpha.shape
    assert polar.cl[1] > 0.3
    assert 1e-5 < polar.cd.mean() < 0.2


def test_short_evolution_improves_or_holds():
    cfg = EvoConfig(
        n_generations=2,
        pop_size=4,
        naca_injection=1,
        model_size="medium",
        seed=1,
        alpha_min=0,
        alpha_max=8,
        alpha_step=2,
    )
    result = run_evolution(cfg)
    assert result.best.fitness > 0
    assert len(result.history) == cfg.n_generations + 1
    assert len(result.population) >= 1
    assert result.best.polar is not None
    evaluate_individual(result.best, cfg)
    assert result.best.thickness < 0.3


def test_evolution_respects_narrow_thickness_band():
    cfg = EvoConfig(
        n_generations=1,
        pop_size=4,
        naca_injection=1,
        model_size="medium",
        seed=3,
        alpha_min=0,
        alpha_max=6,
        alpha_step=2,
        thickness_min=0.08,
        thickness_max=0.10,
    )
    result = run_evolution(cfg)
    assert result.best.fitness > 0
    for ind in result.population:
        assert 0.07 <= ind.thickness <= 0.11
