import numpy as np

from darwin_hor.blade import blade_stl, loft_blade, schmitz_stations
from darwin_hor.config import BladeConfig
from darwin_hor.export import airfoil_dat, airfoil_dxf, airfoil_sldcrv, airfoil_svg
from darwin_hor.geometry import naca4_coordinates


class FakeAirfoil:
    def __init__(self):
        self.name = "NACA4412"
        self.coordinates = naca4_coordinates("4412", n_points=81)


def test_sldcrv_matches_v2_format():
    text = airfoil_sldcrv(FakeAirfoil(), chord_mm=100.0)
    line = text.splitlines()[0]
    assert line.endswith("0mm")
    assert "mm " in line
    x = float(line.split("mm")[0])
    assert 90 < x <= 100.5


def test_dat_dxf_svg_nonempty():
    af = FakeAirfoil()
    assert "NACA4412" in airfoil_dat(af)
    assert "LWPOLYLINE" in airfoil_dxf(af)
    assert "<svg" in airfoil_svg(af)


def test_schmitz_chord_positive_and_decreasing_trend():
    cfg = BladeConfig(radius_m=0.4, hub_radius_m=0.05, n_blades=3, tsr=6.0, n_stations=10)
    stations = schmitz_stations(cfg, cl=0.9, alpha_design_deg=5.0)
    chords = [s.chord for s in stations]
    assert all(c > 0 for c in chords)
    assert stations[0].r == cfg.hub_radius_m
    assert abs(stations[-1].r - cfg.radius_m) < 1e-9
    # root chord larger than tip for typical Schmitz
    assert chords[0] > chords[-1]


def test_stl_has_facets():
    cfg = BladeConfig(n_stations=6, n_airfoil_points=20)
    blade = loft_blade(FakeAirfoil(), cfg, cl=0.8, alpha_design_deg=5.0)
    stl = blade_stl(blade)
    assert stl.startswith("solid")
    assert "facet normal" in stl
    assert blade.faces.shape[0] > 10
    assert blade.vertices.shape[1] == 3
    np.testing.assert_array_less(np.abs(blade.vertices[:, 2].min() - cfg.hub_radius_m), 1e-9)
