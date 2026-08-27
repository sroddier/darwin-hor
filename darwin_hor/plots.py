from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from darwin_hor.blade import Blade
from darwin_hor.evolution import EvolutionResult, GenerationStats
from darwin_hor.geometry import Individual


TEAL = "#3ecf8e"
INK = "#d7e0ea"
GRID = "rgba(255,255,255,0.08)"
PAPER = "#0b1220"
POP_COLORS = ["#3ecf8e", "#62b6ff", "#f4c95d", "#e07a5f", "#c084fc", "#f472b6"]


def _layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(color=INK, family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def airfoil_figure(individuals: list[Individual], title: str = "Profils") -> go.Figure:
    fig = go.Figure()
    for i, ind in enumerate(individuals):
        coords = ind.coords()
        color = POP_COLORS[i % len(POP_COLORS)]
        width = 3 if i == 0 else 1.5
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="lines",
                name=f"{ind.name}  ({ind.fitness:.1f})",
                line=dict(color=color, width=width),
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra>" + ind.name + "</extra>",
            )
        )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title="x/c")
    fig.update_yaxes(title="y/c")
    return _layout(fig, title)


def polar_figure(ind: Individual) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Cl (α)", "Cl/Cd (α)", "Polaire Cl–Cd"),
    )
    if ind.polar is None:
        return _layout(fig, f"Polaire — {ind.name}")
    p = ind.polar
    fig.add_trace(
        go.Scatter(x=p.alpha, y=p.cl, mode="lines+markers", name="Cl", line=dict(color=TEAL)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=p.alpha, y=p.cl_cd, mode="lines+markers", name="Cl/Cd", line=dict(color="#62b6ff")),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=p.cd,
            y=p.cl,
            mode="lines+markers",
            name="Cl–Cd",
            line=dict(color="#f4c95d"),
            text=[f"α={a:.1f}°" for a in p.alpha],
            hovertemplate="%{text}<br>Cd=%{x:.5f}<br>Cl=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="α [°]", row=1, col=1)
    fig.update_yaxes(title_text="Cl", row=1, col=1)
    fig.update_xaxes(title_text="α [°]", row=1, col=2)
    fig.update_yaxes(title_text="Cl/Cd", row=1, col=2)
    fig.update_xaxes(title_text="Cd", row=1, col=3)
    fig.update_yaxes(title_text="Cl", row=1, col=3)
    fig.update_layout(showlegend=False)
    return _layout(fig, f"Polaire — {ind.name}")


def history_figure(result: EvolutionResult) -> go.Figure:
    gens = [h.generation for h in result.history]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gens,
            y=[h.best_fitness for h in result.history],
            mode="lines+markers",
            name="meilleur",
            line=dict(color=TEAL, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gens,
            y=[h.median_fitness for h in result.history],
            mode="lines",
            name="médiane",
            line=dict(color="#62b6ff", width=2, dash="dot"),
        )
    )
    fig.update_xaxes(title="génération")
    fig.update_yaxes(title="fitness")
    return _layout(fig, "Évolution du score")


def population_table_rows(population: list[Individual], n: int = 12) -> list[dict]:
    rows = []
    for ind in population[:n]:
        rows.append(
            {
                "profil": ind.name,
                "génération": ind.generation,
                "mutation": ind.mutation,
                "fitness": round(ind.fitness, 2),
                "Cl/Cd moy.": round(ind.cl_cd_mean, 2),
                "Cl/Cd design": round(ind.cl_cd_design, 2),
                "épaisseur %": round(100.0 * ind.thickness, 2),
                "cambrure %": round(100.0 * ind.camber, 2),
                "confiance": round(ind.confidence_mean, 3),
            }
        )
    return rows


def blade_3d_figure(blade: Blade) -> go.Figure:
    v = blade.vertices
    f = blade.faces
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color=TEAL,
                opacity=0.92,
                name="pale",
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="r [m]",
            aspectmode="data",
            bgcolor=PAPER,
            xaxis=dict(gridcolor=GRID, backgroundcolor=PAPER, color=INK),
            yaxis=dict(gridcolor=GRID, backgroundcolor=PAPER, color=INK),
            zaxis=dict(gridcolor=GRID, backgroundcolor=PAPER, color=INK),
        )
    )
    return _layout(fig, "Pale loftée (BEM Schmitz)")


def planform_figure(blade: Blade) -> go.Figure:
    r = [s.r for s in blade.stations]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Corde", "Vrillage"))
    fig.add_trace(
        go.Scatter(x=r, y=[s.chord * 1000 for s in blade.stations], mode="lines+markers", name="corde", line=dict(color=TEAL)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=r,
            y=[s.twist_deg for s in blade.stations],
            mode="lines+markers",
            name="twist",
            line=dict(color="#f4c95d"),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="r [m]", row=1, col=1)
    fig.update_yaxes(title_text="corde [mm]", row=1, col=1)
    fig.update_xaxes(title_text="r [m]", row=1, col=2)
    fig.update_yaxes(title_text="vrillage [°]", row=1, col=2)
    fig.update_layout(showlegend=False)
    return _layout(fig, "Plan de pale Schmitz")


def overlay_imported(champion: Individual, imported_coords: np.ndarray, imported_name: str) -> go.Figure:
    fig = airfoil_figure([champion], title="Comparaison")
    fig.add_trace(
        go.Scatter(
            x=imported_coords[:, 0],
            y=imported_coords[:, 1],
            mode="lines",
            name=imported_name,
            line=dict(color="#f4c95d", width=2, dash="dash"),
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def live_generation_figure(stats: GenerationStats) -> go.Figure:
    top = stats.population[:6]
    return airfoil_figure(top, title=f"Génération {stats.generation} — {stats.n_alive} profils")
