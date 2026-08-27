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


def _section_xyz_m(blade, index: int) -> np.ndarray:
    n = blade.n_section_pts
    return np.asarray(blade.vertices[index * n : (index + 1) * n], dtype=float)


def station_sldcrv_3d(points_xyz_m: np.ndarray) -> str:
    """Courbe 3D SolidWorks (.sldcrv), coordonnées en mm, profil déjà vrillé et posé à r."""
    pts = np.vstack((points_xyz_m, points_xyz_m[0]))
    lines = [
        f"{x * 1000:.6f}mm {y * 1000:.6f}mm {z * 1000:.6f}mm" for x, y, z in pts
    ]
    return "\n".join(lines) + "\n"


def stations_sldcrv_zip(blade) -> bytes:
    """ZIP d'une courbe .sldcrv par station (insertion Courbe par points XYZ)."""
    import io
    import zipfile

    buf = io.BytesIO()
    readme = (
        "Darwin HOR 3 — courbes de stations (mm)\n"
        "SolidWorks : Insertion > Courbe > Courbe par points XYZ, ou glisser le .sldcrv.\n"
        "Les courbes sont deja a l'echelle, vrillees, a la cote r (axe Z = envergure).\n"
        "Unites : millimetres. Prefere le macro .swb pour avoir plans + esquisses 2D.\n"
    )
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("LIRE_MOI.txt", readme)
        for i, st in enumerate(blade.stations):
            pts = _section_xyz_m(blade, i)
            fname = f"station_{i + 1:02d}_r{st.r * 1000:.0f}mm.sldcrv"
            zf.writestr(fname, station_sldcrv_3d(pts))
    return buf.getvalue()


def _vba_draw_station_sub(index: int, st, pts: np.ndarray, n_stations: int) -> list[str]:
    """Un Sub par station : VBA refuse un Sub de plus de ~64 ko (erreur 'procedure trop grande')."""
    z = float(st.r)
    name_plan = f"Plan_S{index:02d}_r{st.r * 1000:.0f}mm"
    name_sk = f"Esquisse_S{index:02d}"
    n = len(pts)
    sub = f"DrawStation_{index:02d}"
    lines = [
        f"' Station {index}/{n_stations}  r={st.r * 1000:.1f} mm"
        f"  corde={st.chord * 1000:.1f} mm  vrillage={st.twist_deg:.2f} deg",
        f"Sub {sub}(Part As Object)",
        "    Dim ok As Boolean",
        "    Dim feat As Object",
        "    Dim planeFeat As Object",
        "    Part.ClearSelection2 True",
        "    ok = Part.Extension.SelectByID2(\"Plan de face\", \"PLANE\", 0, 0, 0, False, 0, Nothing, 0)",
        "    If Not ok Then ok = Part.Extension.SelectByID2(\"Front Plane\", \"PLANE\", 0, 0, 0, False, 0, Nothing, 0)",
        "    If Not ok Then ok = Part.Extension.SelectByID2(\"Face Plane\", \"PLANE\", 0, 0, 0, False, 0, Nothing, 0)",
        "    Set planeFeat = Nothing",
        "    On Error Resume Next",
        f"    Set planeFeat = Part.FeatureManager.InsertRefPlane(8, {z:.8f}, 0, 0, 0, 0)",
        f"    If planeFeat Is Nothing Then Set planeFeat = Part.FeatureManager.InsertRefPlane(4, {z:.8f}, 0, 0, 0, 0)",
        f"    If planeFeat Is Nothing Then Set planeFeat = Part.FeatureManager.InsertRefPlane(16, {z:.8f}, 0, 0, 0, 0)",
        "    On Error GoTo 0",
        "    If planeFeat Is Nothing Then",
        f'        MsgBox "Echec creation du plan station {index} (API InsertRefPlane)."',
        "        Exit Sub",
        "    End If",
        f'    planeFeat.Name = "{name_plan}"',
        "    Part.ClearSelection2 True",
        f'    ok = Part.Extension.SelectByID2("{name_plan}", "PLANE", 0, 0, 0, False, 0, Nothing, 0)',
        "    Part.SketchManager.InsertSketch True",
        "    Part.SketchManager.AddToDB True",
    ]
    for k in range(n):
        x1, y1 = float(pts[k, 0]), float(pts[k, 1])
        x2, y2 = float(pts[(k + 1) % n, 0]), float(pts[(k + 1) % n, 1])
        lines.append(
            f"    Call Part.SketchManager.CreateLine({x1:.8f}, {y1:.8f}, 0#, {x2:.8f}, {y2:.8f}, 0#)"
        )
    lines += [
        "    Part.SketchManager.AddToDB False",
        "    Part.SketchManager.InsertSketch True",
        "    On Error Resume Next",
        "    Set feat = Part.FeatureByPositionReverse(0)",
        f'    If Not feat Is Nothing Then feat.Name = "{name_sk}"',
        "    On Error GoTo 0",
        "    Part.ClearSelection2 True",
        "End Sub",
        "",
    ]
    return lines


def stations_solidworks_macro(blade, title: str = "DarwinHOR") -> str:
    """Macro VBA (.swb) : nouvelle piece, un plan + une esquisse fermee par station.

    Un Sub VBA ne peut pas depasser ~64 ko : chaque station a son propre Sub.
    Unites API : metres. Affichage : modele de piece par defaut (souvent mm en IUT).
    """
    safe = "".join(ch if ch.isalnum() else "_" for ch in title)[:24] or "DarwinHOR"
    n_st = len(blade.stations)
    lines: list[str] = [
        "' Darwin HOR 3 — plans et esquisses des stations (pas de volume)",
        "' SolidWorks : Outils > Macro > Executer > ce fichier .swb",
        f"' Piece : {safe}  |  {n_st} stations",
        "' Z = envergure (m), esquisse XY = profil corde x vrillage, axe de calage a 25% de corde",
        "' Chaque station = un Sub (limite VBA 64 ko par procedure)",
        "Option Explicit",
        "",
        "Sub main()",
        "    Dim swApp As Object",
        "    Dim Part As Object",
        "    Set swApp = Application.SldWorks",
        "    If swApp Is Nothing Then",
        '        MsgBox "Ouvre SolidWorks, puis : Outils > Macro > Executer."',
        "        Exit Sub",
        "    End If",
        "    Set Part = swApp.NewPart",
        "    If Part Is Nothing Then",
        '        MsgBox "Impossible de creer une piece (modele .prtdot par defaut manquant)."',
        "        Exit Sub",
        "    End If",
        "    On Error Resume Next",
        "    Part.SetAddToDB True",
        "    Part.SetDisplayWhenAdded False",
        "    On Error GoTo 0",
        "",
    ]
    for i in range(n_st):
        lines.append(f"    Call DrawStation_{i + 1:02d}(Part)")
    lines += [
        "",
        "    On Error Resume Next",
        "    Part.SetDisplayWhenAdded True",
        "    Part.SetAddToDB False",
        "    Part.ViewZoomtofit2",
        "    On Error GoTo 0",
        f'    MsgBox "{n_st} stations : plans + esquisses fermees (pas de lissage)." '
        f'& vbCrLf & "Pour la pale : Insertion > Bossage/Base > Lissage, selectionne Esquisse_S01, S02, ... dans l\'ordre."',
        "End Sub",
        "",
    ]
    for i, st in enumerate(blade.stations):
        pts = _section_xyz_m(blade, i)
        lines += _vba_draw_station_sub(i + 1, st, pts, n_st)
    return "\r\n".join(lines)
