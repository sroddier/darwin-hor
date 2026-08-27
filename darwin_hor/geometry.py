from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import aerosandbox as asb
except ImportError:  # pragma: no cover - used only when AeroSandbox is present
    asb = None


def naca4_coordinates(code: str, n_points: int = 121, closed_te: bool = True) -> np.ndarray:
    """Coordonnées Selig (TE extrados → LE → TE intrados) d'un NACA 4 chiffres."""
    code = code.upper().replace("NACA", "").replace(" ", "").strip()
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Code NACA 4 chiffres invalide: {code!r}")

    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    if m > 0 and p == 0:
        p = 0.1

    beta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))
    a4 = -0.1036 if closed_te else -0.1015
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        + a4 * x**4
    )

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m > 0 and p > 0:
        mask = x < p
        yc[mask] = (m / p**2) * (2.0 * p * x[mask] - x[mask] ** 2)
        yc[~mask] = (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x[~mask] - x[~mask] ** 2)
        dyc[mask] = (2.0 * m / p**2) * (p - x[mask])
        dyc[~mask] = (2.0 * m / (1.0 - p) ** 2) * (p - x[~mask])

    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper = np.column_stack((xu[::-1], yu[::-1]))
    lower = np.column_stack((xl[1:], yl[1:]))
    return np.vstack((upper, lower))


def naca4(code: str, n_points: int = 121):
    """Airfoil AeroSandbox à partir d'un code NACA 4 chiffres."""
    if asb is None:
        raise ImportError("aerosandbox est requis")
    coords = naca4_coordinates(code, n_points=n_points)
    name = code.upper() if code.upper().startswith("NACA") else f"NACA{code}"
    return asb.Airfoil(name=name, coordinates=coords)


def naca_thickness_digits(thickness_min: float, thickness_max: float) -> tuple[int, int]:
    """Bornes XX du NACA 4 chiffres correspondant à t/c min–max (en fraction de corde)."""
    lo = int(np.floor(thickness_min * 100.0))
    hi = int(np.ceil(thickness_max * 100.0))
    lo = int(np.clip(lo, 4, 40))
    hi = int(np.clip(hi, 4, 40))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def random_naca4(
    rng: np.random.Generator,
    n_points: int = 121,
    thickness_min: float = 0.08,
    thickness_max: float = 0.22,
):
    """NACA 4 chiffres aléatoire dont l'épaisseur XX entre dans [min, max]."""
    m = int(rng.integers(0, 10))
    p = int(rng.integers(0, 10))
    xx_lo, xx_hi = naca_thickness_digits(thickness_min, thickness_max)
    xx = int(xx_lo if xx_lo == xx_hi else rng.integers(xx_lo, xx_hi + 1))
    if m > 0 and p == 0:
        p = 1
    code = f"{m}{p}{xx:02d}"
    return naca4(code, n_points=n_points)


def parse_dat(text: str) -> tuple[str, np.ndarray]:
    """Lit un .dat Selig / XFOIL / .g1 (1re ligne nom, puis x y)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Fichier profil vide")

    name = "imported"
    start = 0
    first = lines[0].replace(",", " ").split()
    try:
        float(first[0])
        float(first[1])
    except (ValueError, IndexError):
        name = lines[0]
        start = 1

    pts = []
    for ln in lines[start:]:
        parts = ln.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    coords = np.asarray(pts, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 8:
        raise ValueError("Pas assez de points dans le profil")
    return name, coords


def load_dat(text: str, n_points: int = 121):
    if asb is None:
        raise ImportError("aerosandbox est requis")
    name, coords = parse_dat(text)
    af = asb.Airfoil(name=name, coordinates=coords)
    return af.to_kulfan_airfoil()


def as_kulfan(airfoil, name: Optional[str] = None):
    k = airfoil.to_kulfan_airfoil() if hasattr(airfoil, "to_kulfan_airfoil") else airfoil
    if name:
        k.name = name
    return k


def coordinates_of(airfoil) -> np.ndarray:
    return np.asarray(airfoil.coordinates, dtype=float)


def max_thickness(airfoil) -> float:
    return float(airfoil.max_thickness())


def max_camber(airfoil) -> float:
    return float(airfoil.max_camber())


def apply_te_gap(airfoil, thickness: float, name: str):
    """Équivalent GDES TGAP : épaisseur de bord de fuite (fraction de corde)."""
    return as_kulfan(airfoil.set_TE_thickness(float(thickness)), name=name)


def apply_le_radius(airfoil, scale: float, name: str):
    """Équivalent GDES LERA : mise à l'échelle du poids de bord d'attaque CST."""
    k = as_kulfan(airfoil)
    le = float(k.leading_edge_weight) * float(scale)
    return asb.KulfanAirfoil(
        name=name,
        upper_weights=np.array(k.upper_weights, dtype=float).copy(),
        lower_weights=np.array(k.lower_weights, dtype=float).copy(),
        leading_edge_weight=le,
        TE_thickness=float(k.TE_thickness),
    )


def apply_flap(airfoil, hinge_xc: float, deflection_deg: float, name: str):
    """Équivalent GDES FLAP : volet de bord de fuite (déflection >0 vers le bas)."""
    hinged = airfoil.add_control_surface(
        deflection=float(deflection_deg),
        hinge_point_x=float(hinge_xc),
    )
    return as_kulfan(hinged, name=name)


def interpolate_airfoils(a, b, fraction: float, name: str):
    """Croisement XFOIL INTER. `fraction` = part du profil b (0 = a, 1 = b).

    Dans Darwin_HORV2.sci le ratio tiré dans [0.4, 0.6] était écrit avec %d,
    donc tronqué à 0 : le croisement ne mélangeait pas. Ici le flottant est
    conservé.
    """
    blended = a.blend_with_another_airfoil(b, blend_fraction=float(fraction))
    return as_kulfan(blended, name=name)


def mutate_cst(airfoil, rng: np.random.Generator, sigma: float, name: str):
    """Mutation gaussienne des 18 paramètres Kulfan (CST)."""
    k = as_kulfan(airfoil)
    upper = np.array(k.upper_weights, dtype=float) + rng.normal(0.0, sigma, size=8)
    lower = np.array(k.lower_weights, dtype=float) + rng.normal(0.0, sigma, size=8)
    le = float(k.leading_edge_weight) + float(rng.normal(0.0, sigma))
    te = max(0.0, float(k.TE_thickness) + float(rng.normal(0.0, 0.25 * sigma)))
    return asb.KulfanAirfoil(
        name=name,
        upper_weights=upper,
        lower_weights=lower,
        leading_edge_weight=le,
        TE_thickness=te,
    )


def geometry_is_sane(
    airfoil,
    thickness_min: float,
    thickness_max: float,
    thickness_tol: float = 0.006,
) -> bool:
    """Filtre forme + épaisseur. `thickness_tol` (0,6 % de corde) absorbe l'écart NACA vs CST."""
    coords = coordinates_of(airfoil)
    if coords[:, 0].max() > 1.08 or coords[:, 0].min() < -0.08:
        return False
    if np.isnan(coords).any():
        return False
    t = max_thickness(airfoil)
    return (thickness_min - thickness_tol) <= t <= (thickness_max + thickness_tol)


@dataclass
class Polar:
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    confidence: np.ndarray

    @property
    def cl_cd(self) -> np.ndarray:
        cd = np.maximum(self.cd, 1e-6)
        return self.cl / cd

    def at_alpha(self, alpha: float) -> dict[str, float]:
        i = int(np.argmin(np.abs(self.alpha - alpha)))
        return {
            "alpha": float(self.alpha[i]),
            "cl": float(self.cl[i]),
            "cd": float(self.cd[i]),
            "cm": float(self.cm[i]),
            "cl_cd": float(self.cl_cd[i]),
            "confidence": float(self.confidence[i]),
        }

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha.tolist(),
            "cl": self.cl.tolist(),
            "cd": self.cd.tolist(),
            "cm": self.cm.tolist(),
            "confidence": self.confidence.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Polar":
        return cls(
            alpha=np.asarray(d["alpha"], dtype=float),
            cl=np.asarray(d["cl"], dtype=float),
            cd=np.asarray(d["cd"], dtype=float),
            cm=np.asarray(d["cm"], dtype=float),
            confidence=np.asarray(d["confidence"], dtype=float),
        )


@dataclass
class Individual:
    airfoil: object
    name: str
    generation: int
    mutation: str = "seed"
    parents: tuple[str, ...] = ()
    fitness: float = 0.0
    thickness: float = 0.0
    camber: float = 0.0
    cl_cd_mean: float = 0.0
    cl_cd_design: float = 0.0
    confidence_mean: float = 0.0
    polar: Optional[Polar] = None
    uid: int = 0
    extra: dict = field(default_factory=dict)

    def coords(self) -> np.ndarray:
        return coordinates_of(self.airfoil)

    def snapshot(self) -> dict:
        k = as_kulfan(self.airfoil)
        return {
            "name": self.name,
            "generation": self.generation,
            "mutation": self.mutation,
            "parents": list(self.parents),
            "fitness": self.fitness,
            "thickness": self.thickness,
            "camber": self.camber,
            "cl_cd_mean": self.cl_cd_mean,
            "cl_cd_design": self.cl_cd_design,
            "confidence_mean": self.confidence_mean,
            "uid": self.uid,
            "kulfan": {
                "upper_weights": np.asarray(k.upper_weights, dtype=float).tolist(),
                "lower_weights": np.asarray(k.lower_weights, dtype=float).tolist(),
                "leading_edge_weight": float(k.leading_edge_weight),
                "TE_thickness": float(k.TE_thickness),
            },
            "polar": None if self.polar is None else self.polar.to_dict(),
        }
