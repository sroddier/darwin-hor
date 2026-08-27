import numpy as np

from darwin_hor.geometry import max_thickness, naca4_coordinates, parse_dat, random_naca4


def test_naca4412_closed_and_unit_chord():
    coords = naca4_coordinates("4412", n_points=81)
    assert coords.shape[1] == 2
    assert coords.shape[0] >= 80
    assert coords[0, 0] > 0.9
    assert coords[:, 0].min() < 0.02
    # leading edge near x=0
    i_le = coords[:, 0].argmin()
    assert abs(coords[i_le, 0]) < 0.01


def test_naca_symmetric_0012():
    coords = naca4_coordinates("0012", n_points=61)
    # max |y| about 6% for 12% thick
    assert 0.05 < coords[:, 1].max() < 0.08
    assert -0.08 < coords[:, 1].min() < -0.05


def test_parse_dat_with_header():
    text = "NACA 7513\n  1.000000  0.001\n  0.500000  0.080\n  0.000000  0.000\n  0.500000 -0.040\n  1.000000 -0.001\n"
    # too few points - generate more
    lines = ["NACA TEST"]
    for i in range(20):
        x = 1.0 - i / 19.0
        lines.append(f"{x:.4f} {0.1 * x * (1 - x):.4f}")
    for i in range(1, 20):
        x = i / 19.0
        lines.append(f"{x:.4f} {-0.05 * x * (1 - x):.4f}")
    name, coords = parse_dat("\n".join(lines))
    assert name == "NACA TEST"
    assert coords.shape[0] >= 20


def test_evo_config_accepts_percent_thickness():
    from darwin_hor.config import EvoConfig

    cfg = EvoConfig(thickness_min=8.0, thickness_max=25.0)
    assert abs(cfg.thickness_min - 0.08) < 1e-9
    assert abs(cfg.thickness_max - 0.25) < 1e-9


def test_random_naca_matches_thickness_window():
    rng = np.random.default_rng(0)
    for _ in range(12):
        af = random_naca4(rng, n_points=81, thickness_min=0.08, thickness_max=0.10)
        t = max_thickness(af)
        assert 0.07 <= t <= 0.11
