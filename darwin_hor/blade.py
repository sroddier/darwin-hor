from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from darwin_hor.config import BladeConfig
from darwin_hor.geometry import coordinates_of


@dataclass
class BladeStation:
    r: float
    chord: float
    twist_deg: float
    phi_deg: float
    lambda_r: float


@dataclass
class Blade:
    stations: list[BladeStation]
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3) int
    n_section_pts: int
    n_stations: int


def schmitz_stations(cfg: BladeConfig, cl: float, alpha_design_deg: float) -> list[BladeStation]:
    """Corde et vrillage Schmitz (formule IUT classique).

    φ(r) = (2/3) arctan(1/λ_r)
    c(r) = 8 π r (1 − cos φ) / (B Cl)
    β(r) = φ − α_design
    """
    cl_use = cl if cl > 0.35 else cfg.cl_fallback
    r_hub = cfg.hub_radius_m
    r_tip = cfg.radius_m
    rs = np.linspace(r_hub, r_tip, cfg.n_stations)
    c_max = cfg.chord_max_over_r * r_tip
    stations: list[BladeStation] = []
    for r in rs:
        lam_r = cfg.tsr * r / r_tip
        lam_r = max(float(lam_r), 0.05)
        phi = (2.0 / 3.0) * np.arctan(1.0 / lam_r)
        chord = 8.0 * np.pi * r * (1.0 - np.cos(phi)) / (cfg.n_blades * cl_use)
        chord = float(min(max(chord, 1e-4), c_max))
        twist = float(np.degrees(phi) - alpha_design_deg)
        stations.append(
            BladeStation(
                r=float(r),
                chord=chord,
                twist_deg=twist,
                phi_deg=float(np.degrees(phi)),
                lambda_r=lam_r,
            )
        )
    return stations


def _section_points(coords: np.ndarray, station: BladeStation, pitch_axis_xc: float, n_pts: int) -> np.ndarray:
    # resample to n_pts along the closed polyline
    closed = np.vstack((coords, coords[0]))
    seg = np.sqrt(((closed[1:] - closed[:-1]) ** 2).sum(axis=1))
    s = np.concatenate(([0.0], np.cumsum(seg)))
    s /= s[-1]
    su = np.linspace(0.0, 1.0, n_pts, endpoint=False)
    x = np.interp(su, s, closed[:, 0])
    y = np.interp(su, s, closed[:, 1])
    xy = np.column_stack((x, y)) * station.chord
    xy[:, 0] -= pitch_axis_xc * station.chord
    th = np.radians(station.twist_deg)
    c, s_ = np.cos(th), np.sin(th)
    X = xy[:, 0] * c - xy[:, 1] * s_
    Y = xy[:, 0] * s_ + xy[:, 1] * c
    Z = np.full(n_pts, station.r)
    return np.column_stack((X, Y, Z))


def loft_blade(airfoil, cfg: BladeConfig, cl: float, alpha_design_deg: float) -> Blade:
    stations = schmitz_stations(cfg, cl=cl, alpha_design_deg=alpha_design_deg)
    coords = coordinates_of(airfoil)
    n_pts = cfg.n_airfoil_points
    sections = [_section_points(coords, st, cfg.pitch_axis_xc, n_pts) for st in stations]
    vertices = np.vstack(sections)
    faces: list[tuple[int, int, int]] = []

    def vid(i_st: int, i_pt: int) -> int:
        return i_st * n_pts + (i_pt % n_pts)

    for i in range(len(stations) - 1):
        for k in range(n_pts):
            a = vid(i, k)
            b = vid(i, k + 1)
            c = vid(i + 1, k + 1)
            d = vid(i + 1, k)
            faces.append((a, b, c))
            faces.append((a, c, d))

    # caps: fan from section centroid
    for i_st, reverse in ((0, True), (len(stations) - 1, False)):
        sec = sections[i_st]
        center = sec.mean(axis=0)
        ci = len(vertices)
        vertices = np.vstack((vertices, center.reshape(1, 3)))
        for k in range(n_pts):
            a = vid(i_st, k)
            b = vid(i_st, k + 1)
            if reverse:
                faces.append((ci, b, a))
            else:
                faces.append((ci, a, b))

    return Blade(
        stations=stations,
        vertices=vertices,
        faces=np.asarray(faces, dtype=int),
        n_section_pts=n_pts,
        n_stations=len(stations),
    )


def _facet_normal(p0, p1, p2) -> np.ndarray:
    n = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(n)
    if norm < 1e-18:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def blade_stl(blade: Blade, name: str = "pale") -> str:
    v = blade.vertices
    lines = [f"solid {name}"]
    for i, j, k in blade.faces:
        p0, p1, p2 = v[i], v[j], v[k]
        n = _facet_normal(p0, p1, p2)
        lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        lines.append("    outer loop")
        for p in (p0, p1, p2):
            lines.append(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    return "\n".join(lines) + "\n"


def stations_csv(blade: Blade) -> str:
    lines = ["r_m,chord_m,twist_deg,phi_deg,lambda_r"]
    for st in blade.stations:
        lines.append(
            f"{st.r:.6f},{st.chord:.6f},{st.twist_deg:.4f},{st.phi_deg:.4f},{st.lambda_r:.4f}"
        )
    return "\n".join(lines) + "\n"
