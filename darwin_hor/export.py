from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from darwin_hor.geometry import Individual, coordinates_of


def airfoil_dat(airfoil, name: str | None = None) -> str:
    coords = coordinates_of(airfoil)
    label = name or getattr(airfoil, "name", "airfoil")
    lines = [label]
    for x, y in coords:
        lines.append(f" {x:12.6f} {y:12.6f}")
    return "\n".join(lines) + "\n"


def airfoil_sldcrv(airfoil, chord_mm: float = 100.0) -> str:
    """Courbe SolidWorks, même format que Darwin_HORV2 (`%fmm %fmm 0mm`)."""
    coords = coordinates_of(airfoil)
    lines = []
    for x, y in coords:
        lines.append(f"{x * chord_mm:f}mm {y * chord_mm:f}mm 0mm")
    return "\n".join(lines) + "\n"


def airfoil_svg(airfoil, chord_mm: float = 100.0, name: str = "profil") -> str:
    coords = coordinates_of(airfoil)
    pts = coords * chord_mm
    xs, ys = pts[:, 0], pts[:, 1]
    pad = 4.0
    ymin, ymax = float(ys.min()), float(ys.max())
    height = max(ymax - ymin, 1.0)
    width = float(xs.max() - xs.min())
    # SVG y downwards: flip
    parts = []
    for x, y in pts:
        px = x - float(xs.min()) + pad
        py = (ymax - y) + pad
        parts.append(f"{px:.3f},{py:.3f}")
    vb_w = width + 2 * pad
    vb_h = height + 2 * pad
    d = "M " + " L ".join(parts) + " Z"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vb_w:.2f} {vb_h:.2f}" '
        f'width="{vb_w * 4:.0f}" height="{vb_h * 4:.0f}">\n'
        f'  <title>{name}</title>\n'
        f'  <rect width="100%" height="100%" fill="#0b1220"/>\n'
        f'  <path d="{d}" fill="#1f6f5b" fill-opacity="0.35" '
        f'stroke="#3ecf8e" stroke-width="0.4"/>\n'
        f"</svg>\n"
    )


def airfoil_dxf(airfoil, chord_mm: float = 100.0) -> str:
    coords = coordinates_of(airfoil)
    pts = coords * chord_mm
    closed = np.vstack((pts, pts[0]))
    chunks = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "ENTITIES",
        "0",
        "LWPOLYLINE",
        "8",
        "PROFIL",
        "90",
        str(len(closed)),
        "70",
        "1",
    ]
    for x, y in closed:
        chunks.extend(["10", f"{x:.6f}", "20", f"{y:.6f}"])
    chunks.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(chunks) + "\n"


def campaign_json(result, extra: dict | None = None) -> str:
    cfg = result.config
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": result.run_id,
        "config": {
            "n_generations": cfg.n_generations,
            "pop_size": cfg.pop_size,
            "re": cfg.re,
            "mach": cfg.mach,
            "alpha_min": cfg.alpha_min,
            "alpha_max": cfg.alpha_max,
            "alpha_step": cfg.alpha_step,
            "alpha_design": cfg.design_alpha(),
            "thickness_max": cfg.thickness_max,
            "thickness_min": cfg.thickness_min,
            "model_size": cfg.model_size,
            "fitness_mode": cfg.fitness_mode,
            "seed": cfg.seed,
        },
        "best": result.best.snapshot(),
        "history": [
            {
                "generation": h.generation,
                "n_alive": h.n_alive,
                "best_fitness": h.best_fitness,
                "median_fitness": h.median_fitness,
                "best_name": h.best_name,
                "best_thickness": h.best_thickness,
                "best_cl_cd_mean": h.best_cl_cd_mean,
            }
            for h in result.history
        ],
        "population": [ind.snapshot() for ind in result.population],
        "extra": extra or {},
    }
    return json.dumps(payload, indent=2)


def polar_csv(ind: Individual) -> str:
    if ind.polar is None:
        return "alpha,cl,cd,cm,cl_cd,confidence\n"
    lines = ["alpha,cl,cd,cm,cl_cd,confidence"]
    p = ind.polar
    for i in range(len(p.alpha)):
        lines.append(
            f"{p.alpha[i]:.3f},{p.cl[i]:.6f},{p.cd[i]:.8f},{p.cm[i]:.6f},"
            f"{p.cl_cd[i]:.4f},{p.confidence[i]:.4f}"
        )
    return "\n".join(lines) + "\n"
