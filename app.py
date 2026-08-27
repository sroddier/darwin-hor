"""Darwin HOR 3 — interface Streamlit."""

from __future__ import annotations

import streamlit as st

from darwin_hor.blade import blade_stl, loft_blade, stations_csv
from darwin_hor.config import MODEL_SIZES, BladeConfig, EvoConfig
from darwin_hor.evolution import run_evolution
from darwin_hor.export import (
    airfoil_dat,
    airfoil_dxf,
    airfoil_sldcrv,
    airfoil_svg,
    campaign_json,
    polar_csv,
    stations_sldcrv_zip,
    stations_solidworks_macro,
)
from darwin_hor.geometry import Individual, load_dat, naca4, parse_dat
from darwin_hor.plots import (
    airfoil_figure,
    blade_3d_figure,
    history_figure,
    overlay_imported,
    planform_figure,
    polar_figure,
    population_table_rows,
)
from darwin_hor.solver import evaluate_individual

st.set_page_config(page_title="Darwin HOR 3", page_icon="🌬️", layout="wide")

st.markdown(
    """
<style>
    .block-container { padding-top: 1.2rem; }
    h1 { font-weight: 650; letter-spacing: -0.03em; }
    div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
</style>
""",
    unsafe_allow_html=True,
)


def _hint(text: str) -> None:
    st.sidebar.caption(text)


def _cfg_from_sidebar() -> tuple[EvoConfig, BladeConfig]:
    st.sidebar.title("Darwin HOR 3")
    st.sidebar.caption("Évolution de profils de pale — IUT GIM Saint-Denis")
    with st.sidebar.expander("Comment lire les réglages ?", expanded=False):
        st.markdown(
            "Sous chaque curseur : **à quoi ça sert**. "
            "Le **?** à droite du nom donne un peu plus de détail. "
            "Les valeurs par défaut reprennent le programme Scilab Darwin HOR V2."
        )

    st.sidebar.subheader("Campagne")
    st.sidebar.caption("Combien de temps Darwin cherche, et avec combien de profils.")
    n_generations = st.sidebar.slider(
        "Nombre de générations",
        2,
        80,
        20,
        help=(
            "Un cycle complet : naissances (NACA ou mutants) → calcul de polaire → "
            "sélection des survivants. Plus il y en a, plus Darwin a le temps d’améliorer "
            "le champion, mais le calcul dure plus longtemps. 20 est un bon premier essai."
        ),
    )
    _hint("Tours d’évolution. 20 ≈ un TP ; 50+ si tu cherches un vrai podium.")

    pop_size = st.sidebar.slider(
        "Profils par génération",
        4,
        40,
        12,
        help=(
            "Taille de la population. Beaucoup de profils = plus de diversité "
            "(moins de risque de rester coincé sur un NACA moyen), mais chaque "
            "génération est plus lente. 12 est un compromis cours / concours."
        ),
    )
    _hint("Individus évalués à chaque tour. Plus c’est grand, plus c’est divers — et lent.")

    seed = st.sidebar.number_input(
        "Graine (reproductible)",
        min_value=0,
        value=42,
        step=1,
        help=(
            "Départ du générateur aléatoire. Même graine + mêmes réglages = exactement "
            "la même campagne. Change-la pour explorer un autre hasard, ou garde-la "
            "pour comparer deux binômes à armes égales."
        ),
    )
    _hint("Même nombre = même tirage. Utile pour retracer un champion en compte-rendu.")

    st.sidebar.subheader("Aéro (NeuralFoil)")
    st.sidebar.caption("Conditions de vol 2D dans lesquelles le profil est noté.")
    re = st.sidebar.number_input(
        "Reynolds",
        min_value=1.0e4,
        max_value=1.0e7,
        value=3.0e5,
        step=1.0e4,
        format="%.0f",
        help=(
            "Re = ρ × V × corde / μ. Ça dit si l’écoulement est « visqueux » (petit Re, "
            "beaucoup de traînée) ou plus inertiel. Mini-éolienne de concours : souvent "
            "5×10⁴ à 5×10⁵. La V2 Scilab était figée à 300 000."
        ),
    )
    _hint("Re = ρVc/μ. Trop bas : le profil décroche tôt. 3×10⁵ = valeur V2, réaliste en IUT.")

    mach = st.sidebar.number_input(
        "Mach",
        min_value=0.0,
        max_value=0.6,
        value=0.20,
        step=0.05,
        help=(
            "Mach = vitesse / vitesse du son. Sous 0,3 l’air est quasi incompressible : "
            "ça change très peu le résultat. 0,2 recopie la V2. Sur une éolienne étudiante "
            "le vrai Mach bout de pale est souvent plus bas (~0,05–0,15)."
        ),
    )
    _hint("Vitesse / son. En dessous de 0,3, tu peux le laisser à 0,2 sans t’en soucier.")

    c1, c2 = st.sidebar.columns(2)
    alpha_min = c1.number_input(
        "α min [°]",
        value=0.0,
        step=1.0,
        help=(
            "Plus petit angle d’incidence de la polaire. α = angle entre la corde du profil "
            "et le vent relatif. 0° = vent dans l’axe de la corde."
        ),
    )
    alpha_max = c2.number_input(
        "α max [°]",
        value=11.0,
        step=1.0,
        help=(
            "Plus grand angle d’incidence calculé. Au-delà d’~12–15° un profil épais "
            "décroche (Cl chute, Cd explose). La V2 balayait 0 à 11°."
        ),
    )
    _hint("Plage d’angles d’attaque de la polaire (pas 1°). Darwin note le profil sur toute cette plage.")

    alpha_design = st.sidebar.slider(
        "α design [°]",
        min_value=float(alpha_min),
        max_value=float(max(alpha_min, alpha_max)),
        value=float(0.5 * (alpha_min + alpha_max)),
        help=(
            "Angle de fonctionnement visé. Sert (1) au score « design » = Cl/Cd à cet angle, "
            "(2) au vrillage de pale : β = φ − α_design. Choisis un α où Cl/Cd est bon "
            "sans être au bord du décrochage (souvent 5–8°)."
        ),
    )
    _hint("Angle « de croisière » : Cl pour Schmitz, et Cl/Cd si tu scores en mode design.")

    model_size = st.sidebar.selectbox(
        "Modèle NeuralFoil",
        MODEL_SIZES,
        index=MODEL_SIZES.index("large"),
        help=(
            "Taille du réseau qui imite XFOIL. Plus c’est grand (xlarge…), plus c’est proche "
            "d’XFOIL et plus c’est lent. « large » est le bon défaut. « medium » pour un TP "
            "rapide, « xlarge » pour valider un champion."
        ),
    )
    _hint("Précision du substitut XFOIL. large = défaut. xlarge pour le profil que tu vas usiner.")

    st.sidebar.subheader("Contraintes & fitness")
    st.sidebar.caption("Comment on note un profil, et ce qu’on refuse de fabriquer.")
    t_max_pct = st.sidebar.slider(
        "Épaisseur max [%]",
        8.0,
        30.0,
        22.0,
        0.5,
        help=(
            "Épaisseur maximale relative t/c, en % de la corde. Un profil de corde 100 mm "
            "et 12 % fait 12 mm d’épaisseur. Trop épais = beaucoup de traînée. 22 % recopie "
            "le dialogue V2 (ici le filtre s’applique vraiment)."
        ),
    )
    _hint("t/c max. Au-dessus, le profil est pénalisé / éliminé. 12–18 % est typique d’une pale imprimée.")

    t_min_pct = st.sidebar.slider(
        "Épaisseur min [%]",
        4.0,
        16.0,
        8.0,
        0.5,
        help=(
            "Plancher d’épaisseur. Un profil ultra-fin a souvent un super Cl/Cd sur le papier "
            "mais casse, vrille à l’impression 3D, ou a un bord de fuite infaisable. 8 % "
            "reste usinable / imprimable pour un concours IUT."
        ),
    )
    _hint("Sous ce % le profil est trop fragile (structure, impression). Ne descends pas sous ~8 %.")
    if t_min_pct >= t_max_pct:
        st.sidebar.error("L'épaisseur min doit être inférieure au max.")
        t_min_pct = max(4.0, t_max_pct - 4.0)
    t_max = t_max_pct / 100.0
    t_min = t_min_pct / 100.0

    fitness_mode = st.sidebar.selectbox(
        "Score",
        options=["moyenne", "design", "classique"],
        index=0,
        format_func=lambda x: {
            "moyenne": "Moyenne Cl/Cd sur la plage α",
            "design": "Cl/Cd seulement à α design",
            "classique": "Somme |Cl/Cd| (ancienne V2)",
        }[x],
        help=(
            "Note qui décide qui survit. "
            "Moyenne : finesse moyenne entre α min et α max (robuste, recommandé). "
            "Design : on ne regarde que l’angle de croisière (pointu, risque de mal se tenir ailleurs). "
            "Classique : somme |Cl/Cd| de la V2 Scilab, pour comparer à tes anciens runs."
        ),
    )
    _hint("Cl/Cd = portance / traînée = « finesse ». Plus c’est haut, mieux le profil porte sans freiner.")

    with st.sidebar.expander("Sélection darwinienne", expanded=False):
        st.caption("Règles de survie, comme dans Darwin_HORV2.sci — c’est ça qui garde de la diversité.")
        weak = st.slider(
            "Survie des faibles",
            0.0,
            0.3,
            0.05,
            0.01,
            help=(
                "Un profil sous la médiane a cette probabilité de survivre quand même (défaut 5 %). "
                "Sans ça, Darwin converge trop vite sur un seul NACA et n’explore plus. "
                "C’est volontairement « injuste » : la nature garde quelques originaux."
            ),
        )
        st.caption("Chance qu’un mauvais profil survive. 5 % = défaut V2. 0 % = sélection trop sévère.")

        death = st.slider(
            "Mort aléatoire des bons",
            0.0,
            0.1,
            0.01,
            0.01,
            help=(
                "Même un profil au-dessus de la médiane a 1 % de chances de mourir. "
                "Ça évite qu’une lignée moyenne-mais-nombreuse envahisse tout. "
                "Les élites (ci-dessous) y échappent."
            ),
        )
        st.caption("1 % des « assez bons » meurent par hasard. Ça casse les monocultures.")

        elites = st.slider(
            "Élites protégées",
            1,
            6,
            2,
            help=(
                "Les N meilleurs de la génération sont intouchables : ils ne peuvent pas "
                "mourir au tirage. Sans élitisme, on risque de perdre le champion d’un "
                "coup de malchance. 2 suffit."
            ),
        )
        st.caption("Les N meilleurs sont toujours recopiés à la génération suivante.")

        inject = st.slider(
            "NACA injectés / génération",
            0,
            12,
            4,
            help=(
                "À chaque tour, Darwin fait entrer de nouveaux NACA 4 chiffres au hasard "
                "(comme AleaNaca dans la V2). C’est de l’immigration génétique : ça "
                "réinjecte des formes classiques si la population devient trop mutante."
            ),
        )
        st.caption("Nouveaux NACA « sauvages » à chaque tour, pour ne pas tourner en vase clos.")

    st.sidebar.subheader("Pale 3D (BEM Schmitz)")
    st.sidebar.caption("Ne change pas le profil. Ça enroule le champion en pale imprimable.")
    radius = st.sidebar.number_input(
        "Rayon rotor R [m]",
        0.05,
        5.0,
        0.40,
        0.05,
        help=(
            "Distance axe → bout de pale. Diamètre du rotor = 2R. Mesure le règlement "
            "du concours (souvent un diamètre max). Défaut 0,40 m → rotor Ø 80 cm."
        ),
    )
    _hint("Bout de pale. Ø rotor = 2R. À caler sur le cahier des charges du concours.")

    hub = st.sidebar.number_input(
        "Rayon moyeu [m]",
        0.01,
        1.0,
        0.05,
        0.01,
        help=(
            "Où la pale commence, autour du moyeu / de la génératrice. En dessous, "
            "pas de profil : c’est le moyeu. Doit rester < R. 5 cm convient à une "
            "petite machine."
        ),
    )
    _hint("Début de pale (trou de moyeu). La première station est à ce rayon.")

    n_blades = st.sidebar.slider(
        "Nombre de pales",
        2,
        6,
        3,
        help=(
            "B dans la formule de corde Schmitz : c = 8πr(1−cos φ)/(B Cl). "
            "Plus de pales → corde plus étroite (même portance répartie). "
            "3 est le classique HAWT. 2 = pales plus larges, plus d’à-coups."
        ),
    )
    _hint("Plus de pales = pales plus étroites. 3 est le standard axe horizontal.")

    tsr = st.sidebar.slider(
        "TSR λ",
        2.0,
        12.0,
        6.0,
        0.5,
        help=(
            "Tip Speed Ratio λ = ΩR / Vvent = vitesse du bout de pale / vitesse du vent. "
            "Éolienne lente multi-pales ≈ 2–4 ; HAWT rapide 3 pales ≈ 5–8. "
            "λ grand → pales étroites et peu vrillées. 6 est un bon départ concours."
        ),
    )
    _hint("λ = ΩR/V. 6 ≈ 3 pales rapides. Trop grand : pales filiformes, peu de couple au démarrage.")

    n_stations = st.sidebar.slider(
        "Stations",
        6,
        24,
        12,
        help=(
            "Une station = une coupe de pale à un rayon r (moyeu → bout). "
            "À chaque coupe on calcule corde et vrillage, on pose le profil, "
            "puis on lofte le tout. Ça n’optimise pas le profil : ça dessine la 3D. "
            "12 = lisse et léger. 24 = STL plus fin, plus lourd."
        ),
    )
    _hint("Nombre de coupes le long de la pale. 12 suffit ; augmente si le STL a des facettes.")

    chord_mm = st.sidebar.number_input(
        "Corde export 2D [mm]",
        10.0,
        1000.0,
        100.0,
        10.0,
        help=(
            "Échelle du fichier SolidWorks .sldcrv / DXF / SVG. La V2 exportait une corde "
            "de 100 mm (coordonnées × 100). Ça ne change pas le calcul aéro, seulement "
            "la taille de la courbe que tu ouvres dans le CAD."
        ),
    )
    _hint("Taille de la courbe 2D pour SolidWorks/DXF. 100 mm = identique à la V2.")

    evo = EvoConfig(
        n_generations=int(n_generations),
        pop_size=int(pop_size),
        re=float(re),
        mach=float(mach),
        alpha_min=float(alpha_min),
        alpha_max=float(alpha_max),
        alpha_design=float(alpha_design),
        thickness_max=float(t_max),
        thickness_min=float(t_min),
        model_size=str(model_size),
        fitness_mode=str(fitness_mode),
        weak_survival=float(weak),
        elite_death=float(death),
        n_elites=int(elites),
        naca_injection=int(inject),
        seed=int(seed),
        chord_export_mm=float(chord_mm),
    )
    blade = BladeConfig(
        radius_m=float(radius),
        hub_radius_m=float(hub),
        n_blades=int(n_blades),
        tsr=float(tsr),
        n_stations=int(n_stations),
    )
    return evo, blade


def _ensure_state():
    if "result" not in st.session_state:
        st.session_state.result = None


@st.cache_resource(show_spinner="Chargement du démonstrateur NACA 4412…")
def _demo_individual_cached() -> Individual:
    cfg = EvoConfig(model_size="medium", alpha_min=0.0, alpha_max=11.0, alpha_step=1.0)
    af = naca4("4412", n_points=cfg.n_points)
    ind = Individual(airfoil=af, name="NACA4412", generation=0, mutation="demo", uid=0)
    return evaluate_individual(ind, cfg)


def _downloads(ind: Individual, cfg: EvoConfig, result=None):
    stem = ind.name.replace(" ", "_")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Profil .dat (XFOIL/Selig)", airfoil_dat(ind.airfoil, ind.name), f"{stem}.dat", key="dl_dat")
    c2.download_button("SolidWorks .sldcrv", airfoil_sldcrv(ind.airfoil, cfg.chord_export_mm), f"{stem}.sldcrv", key="dl_sld")
    c3.download_button("DXF", airfoil_dxf(ind.airfoil, cfg.chord_export_mm), f"{stem}.dxf", key="dl_dxf")
    c4.download_button("SVG", airfoil_svg(ind.airfoil, cfg.chord_export_mm, ind.name), f"{stem}.svg", key="dl_svg")
    p1, p2 = st.columns(2)
    p1.download_button("Polaire CSV", polar_csv(ind), f"{stem}_polaire.csv", key="dl_csv")
    if result is not None:
        p2.download_button("Campagne JSON", campaign_json(result), f"darwin_run{result.run_id}.json", key="dl_json")


def main():
    _ensure_state()
    cfg, blade_cfg = _cfg_from_sidebar()

    st.title("Darwin HOR 3")
    st.caption(
        "Algorithme génétique de profils 2D pour pale d’éolienne à axe horizontal. "
        "Reprend Darwin_HORV2.sci (Scilab + XFOIL) avec NeuralFoil, sans binaire Windows. "
        "Tous les réglages de gauche ont une explication — survole le **?** ou lis la ligne grise en dessous."
    )

    run = st.button(
        "Lancer l’évolution",
        type="primary",
        use_container_width=False,
        help=(
            "Démarre Darwin avec les réglages du panneau gauche. "
            "Chaque génération mute les survivants (INTER, bord de fuite, bord d’attaque, volet, CST), "
            "calcule une polaire NeuralFoil, puis tue les profils trop épais ou trop nuls."
        ),
    )
    n_eval = cfg.n_generations * (cfg.pop_size + cfg.naca_injection)
    st.caption(
        f"{cfg.n_generations} générations × ~{cfg.pop_size} individus  "
        f"· Re = {cfg.re:.0f}  · α = {cfg.alpha_min:.0f}…{cfg.alpha_max:.0f}°  "
        f"· ~{n_eval} polaires NeuralFoil"
    )

    progress = st.progress(0.0, text="En attente")
    live_plot = st.empty()

    if run:
        status_box = st.empty()

        def on_gen(stats):
            frac = (stats.generation) / max(cfg.n_generations, 1)
            progress.progress(
                min(1.0, frac),
                text=(
                    f"Génération {stats.generation}/{cfg.n_generations} — "
                    f"meilleur {stats.best_fitness:.1f}  ({stats.best_name})"
                ),
            )
            live_plot.plotly_chart(
                airfoil_figure(stats.population[:5], title=f"Génération {stats.generation}"),
                use_container_width=True,
            )
            status_box.info(
                f"{stats.n_alive} survivants · médiane {stats.median_fitness:.1f} · "
                f"épaisseur du meilleur {100 * stats.best_thickness:.1f} %"
            )

        try:
            with st.spinner("Évolution en cours…"):
                st.session_state.result = run_evolution(cfg, on_generation=on_gen)
            progress.progress(1.0, text="Terminé")
        except RuntimeError as exc:
            progress.progress(0.0, text="Échec")
            st.error(str(exc))
            st.info(
                "Le filtre d'épaisseur s'applique dès la 1re génération. "
                "Avec min = 8 % et max = 22 % (défaut V2), ça passe. "
                "Un max trop bas (ex. 8 %) refuse tous les NACA, qui sont typiquement épais de 10 à 21 %."
            )
            return

    result = st.session_state.result
    if result is None:
        demo = _demo_individual_cached()
        st.info("Aucun run encore. Aperçu d’un NACA 4412 ci-dessous — clique **Lancer l’évolution** pour Darwin.")
        st.plotly_chart(airfoil_figure([demo], title="NACA 4412 (démo)"), use_container_width=True)
        st.plotly_chart(polar_figure(demo), use_container_width=True)
        return

    best = result.best
    tabs = st.tabs(["Évolution", "Champion", "Polaire", "Pale 3D", "Comparer", "À propos"])

    with tabs[0]:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Fitness champion",
            f"{best.fitness:.1f}",
            help="Note de classement (selon le mode Score). C’est elle qui décide qui survit, pas le Cl tout seul.",
        )
        m2.metric(
            "Cl/Cd moyen",
            f"{best.cl_cd_mean:.1f}",
            help="Finesse moyenne sur la plage α : portance / traînée. 80–120 est déjà très bon pour une mini-éolienne.",
        )
        m3.metric(
            "Épaisseur",
            f"{100 * best.thickness:.1f} %",
            help="t/c du champion. 10–18 % : bon compromis aéro / fabrication. Au-delà de 22 % Darwin le refuse.",
        )
        m4.metric(
            "Profils finaux",
            f"{len(result.population)}",
            help="Survivants de la dernière génération (après sélection). Les 6 meilleurs sont dessinés ci-dessous.",
        )
        st.plotly_chart(history_figure(result), use_container_width=True)
        st.plotly_chart(
            airfoil_figure(result.population[:6], title="Population finale (6 meilleurs)"),
            use_container_width=True,
        )
        st.caption(
            "Tableau : **mutation** = opérateur qui a créé le profil (naca, inter, tegap, lerad, flap, cst) · "
            "**fitness** = note de survie · **confiance** = fiabilité NeuralFoil (vise > 0,7)."
        )
        st.dataframe(population_table_rows(result.population), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader(best.name)
        st.caption(
            f"mutation {best.mutation} · génération {best.generation} · "
            f"parents : {', '.join(best.parents) if best.parents else '—'}"
        )
        st.plotly_chart(airfoil_figure([best], title="Champion"), use_container_width=True)
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Cl/Cd @ α design",
            f"{best.cl_cd_design:.1f}",
            help="Finesse à l’angle de croisière choisi à gauche. C’est le Cl/Cd que Schmitz utilise indirectement via Cl.",
        )
        k2.metric(
            "Cambrure",
            f"{100 * best.camber:.2f} %",
            help="Flèche max de la ligne moyenne, en % de corde. Plus c’est cambré, plus ça porte (et plus ça décroche tôt).",
        )
        k3.metric(
            "Confiance NeuralFoil",
            f"{best.confidence_mean:.2f}",
            help="1 ≈ le réseau est dans son domaine (proche XFOIL). < 0,5 : profil bizarre, méfie-toi du score, revalide dans XFOIL.",
        )
        st.caption(
            "Exports : **.dat** pour XFOIL / XFLR5 · **.sldcrv** courbe SolidWorks (corde du réglage « export 2D ») · "
            "**DXF** laser/CNC · **SVG** dessin · **JSON** pour retracer toute la campagne."
        )
        _downloads(best, result.config, result)

    with tabs[2]:
        st.caption(
            "**Cl** = coefficient de portance (ça « soulève » la pale) · "
            "**Cd** = coefficient de traînée (ça freine) · "
            "**Cl/Cd** = finesse, à maximiser · "
            "**α** = angle d’attaque. La polaire Cl–Cd se lit comme un nuage : on veut haut et à gauche."
        )
        st.plotly_chart(polar_figure(best), use_container_width=True)
        if best.polar is not None:
            st.dataframe(
                {
                    "α": best.polar.alpha,
                    "Cl": best.polar.cl.round(4),
                    "Cd": best.polar.cd.round(6),
                    "Cl/Cd": best.polar.cl_cd.round(2),
                    "confiance": best.polar.confidence.round(3),
                },
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
        cl = best.polar.at_alpha(result.config.design_alpha())["cl"] if best.polar is not None else 0.8
        blade = loft_blade(best.airfoil, blade_cfg, cl=float(cl), alpha_design_deg=result.config.design_alpha())
        st.caption(
            f"BEM Schmitz · Cl utilisé = {cl:.2f} @ α = {result.config.design_alpha():.1f}° · "
            f"R = {blade_cfg.radius_m:.2f} m · λ = {blade_cfg.tsr:.1f} · {blade_cfg.n_blades} pales. "
            "Une **station** est une coupe à un rayon r : on y calcule corde et vrillage, puis on lofte."
        )
        st.plotly_chart(planform_figure(blade), use_container_width=True)
        st.plotly_chart(blade_3d_figure(blade), use_container_width=True)
        b1, b2, b3, b4 = st.columns(4)
        stem = best.name.replace(" ", "_")
        b1.download_button(
            "Pale STL",
            blade_stl(blade, name=stem),
            f"{stem}_pale.stl",
            key="dl_stl",
        )
        b2.download_button("Stations CSV", stations_csv(blade), f"{stem}_stations.csv", key="dl_stations")
        b3.download_button(
            "SolidWorks plans + esquisses (.swb)",
            stations_solidworks_macro(blade, title=stem),
            f"{stem}_stations.swb",
            key="dl_swb",
            help=(
                "Macro VBA : Outils > Macro > Exécuter dans SolidWorks. "
                "Crée une pièce neuve avec un plan et une esquisse fermée par station, sans lissage."
            ),
        )
        b4.download_button(
            "Courbes stations .sldcrv (zip)",
            stations_sldcrv_zip(blade),
            f"{stem}_stations_sldcrv.zip",
            key="dl_sldcrv_zip",
            mime="application/zip",
            help="Une courbe 3D .sldcrv par station (mm), déjà vrillée et placée le long de Z.",
        )
        with st.expander("Procédure SolidWorks — plans et esquisses", expanded=True):
            st.markdown(
                """
### But

Obtenir une **pièce SolidWorks qui ne contient que** :
- un **plan** par station (`Plan_S01`, `Plan_S02`, …) perpendiculaire à l’envergure ;
- une **esquisse fermée** sur ce plan (`Esquisse_S01`, …) : le profil déjà **mis à l’échelle**, **vrillé** et posé à la cote **r**.

Pas de volume : le lissage se fait **à la main** ensuite (contrôle pédagogique).

### A. Préparer le fichier

1. Dans Darwin, lance une évolution puis ouvre cet onglet **Pale 3D**.
2. Clique **SolidWorks plans + esquisses (.swb)** et enregistre le fichier (ex. `NACA4412_stations.swb`).
3. Ouvre **SolidWorks** (une pièce ou rien du tout : le macro crée une pièce neuve).

### B. Lancer le macro

4. Menu **Outils → Macro → Exécuter** (pas « Nouveau »).
5. Type de fichier : **Macro SolidWorks (\\*.swb, \\*.swp)** si tu ne vois pas le `.swb`.
6. Choisis le fichier téléchargé → **Ouvrir**.
7. Si SolidWorks bloque : **Outils → Options → SolidWorks → Macro** → autoriser l’exécution, puis recommence.

### C. Vérifier la pièce créée

8. L’arbre doit montrer, dans l’ordre : `Plan_S01` + `Esquisse_S01`, puis S02, etc. jusqu’au bout de pale.
9. **Zoom tout** : tu dois voir les profils empilés le long de Z.
10. **Unités** : le macro travaille en **mètres**. Avec un modèle IUT en mm, un rayon R = 0,40 m s’affiche **400 mm**. Si tout est 1000 fois trop petit, le document est en mètres : **Outils → Options → Propriétés du document → Unités → MMGS**, ou échelle × 1000.

Chaque esquisse est dans le **plan de la station** (Z = r), origine ≈ **quart avant** (axe de calage à 25 % de corde).

### D. Faire la pale (lissage)

11. **Insertion → Bossage/Base → Lissage** (Loft).
12. Dans **Profils**, clique les esquisses **dans l’ordre** : `Esquisse_S01` (moyeu) → `Esquisse_S02` → … → dernière (bout).
13. Coche **Fusionner les résultats** si tu veux un seul solide. Valide (coche verte).
14. Contrôle : longueur de pale ≈ **R − rayon moyeu** (défaut 400 − 50 = **350 mm**).

### Si le macro refuse de partir (plan B)

15. Télécharge **Courbes stations .sldcrv (zip)** et décompresse.
16. **Insertion → Courbe → Courbe par points XYZ** (ou glisser un `.sldcrv` dans la zone graphique).
17. Un fichier = une station, coordonnées **déjà en mm** dans l’espace 3D. Ensuite lissage sur ces courbes, ou projette chaque courbe sur un plan pour retrouver des esquisses 2D.
                """
            )
        st.markdown(
            """
- **r** : distance à l’axe (moyeu → bout)
- **corde** : largeur du profil à cette coupe (plus large à l’emplanture)
- **vrillage β** : rotation du profil pour que l’angle d’attaque reste ≈ α design (β = φ − α)
- **φ** : angle du vent relatif (inflow Schmitz)
            """
        )
        st.dataframe(
            {
                "r [m]": [s.r for s in blade.stations],
                "corde [mm]": [round(s.chord * 1000, 1) for s in blade.stations],
                "vrillage [°]": [round(s.twist_deg, 2) for s in blade.stations],
                "φ [°]": [round(s.phi_deg, 2) for s in blade.stations],
            },
            use_container_width=True,
            hide_index=True,
        )

    with tabs[4]:
        uploaded = st.file_uploader(
            "Importer un .dat / .g1 (profil Darwin V2, XFOIL ou UIUC) pour le superposer au champion",
            type=["dat", "g1", "txt"],
            help=(
                "Fichier Selig/XFOIL : une ligne de nom, puis des paires x y (corde unitaire). "
                "Les .g1 de Darwin V2 et les « Le meilleurNACA….dat » marchent. "
                "On redessine le profil et on recalcule sa polaire avec les mêmes Re / α que le run."
            ),
        )
        if uploaded is not None:
            text = uploaded.read().decode("utf-8", errors="ignore")
            name, coords = parse_dat(text)
            st.plotly_chart(overlay_imported(best, coords, name), use_container_width=True)
            try:
                imported = Individual(airfoil=load_dat(text), name=name, generation=0, mutation="import")
                imported = evaluate_individual(imported, result.config)
                c1, c2, c3 = st.columns(3)
                c1.metric("Champion Cl/Cd", f"{best.cl_cd_mean:.1f}")
                c2.metric(f"{name} Cl/Cd", f"{imported.cl_cd_mean:.1f}")
                c3.metric("Épaisseur importé", f"{100 * imported.thickness:.1f} %")
            except Exception as exc:
                st.warning(f"Polaire de l’import impossible : {exc}")
        else:
            st.caption("Astuce : tu peux coller un `NACA7513.g1` ou `Le meilleurNACA7513.dat` de la V2 Scilab.")

    with tabs[5]:
        st.markdown(
            """
**Darwin HOR 3** reprend le programme Scilab `Darwin_HORV2.sci` (Stéphane Roddier, IUT GIM Saint-Denis).

| V2 Scilab | V3 |
|---|---|
| `xfoil.exe` + fichiers `.g1` / `.dat.darwin` | NeuralFoil en mémoire, sans binaire |
| NACA 4 chiffres aléatoires | identique |
| INTER, TGAP, LERA, FLAP | identique, plus mutation CST |
| Croisement `%d` → ratio 0 (bug) | interpolation réelle 40–60 % |
| Fitness = Σ \|Cl/Cd\| | moyenne pondérée par la confiance, ou mode classique |
| Épaisseur max mal filtrée | 22 % = 22 % de corde |
| Export `.sldcrv` 100 mm | `.sldcrv`, `.dat`, DXF, SVG, STL pale, **macro SolidWorks plans+esquisses** |

Aéro : [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) (Peter Sharpe, MIT), entraîné sur des millions de polaires XFOIL.
La pale 3D utilise les formules de **Schmitz** :  
φ = (2/3) arctan(1/λ_r),  c = 8π r (1−cos φ) / (B Cl),  β = φ − α_design.

### Procédure SolidWorks (rappel)

1. Onglet **Pale 3D** → télécharger **SolidWorks plans + esquisses (.swb)**.
2. SolidWorks : **Outils → Macro → Exécuter** → le `.swb` (autoriser les macros si demandé).
3. Contrôler l’arbre : `Plan_S01` + `Esquisse_S01`, puis S02… (profils vrillés, à l’échelle, à la cote r).
4. Unités : API en **mètres** → en MMGS, R = 0,40 m s’affiche **400 mm**.
5. **Insertion → Bossage/Base → Lissage**, profils dans l’ordre S01 (moyeu) → bout.
6. Plan B : ZIP `.sldcrv` → **Insertion → Courbe → Courbe par points XYZ** (mm, déjà placées en 3D).

### Mini-glossaire

| Mot | Sens rapide |
|---|---|
| **Cl** | Coefficient de portance. Plus c’est haut, plus la pale « pousse » dans le sens utile. |
| **Cd** | Coefficient de traînée. Plus c’est bas, moins ça freine. |
| **Cl/Cd** | Finesse. C’est le rapport qu’on cherche à maximiser. |
| **α** | Angle d’attaque : corde du profil vs vent relatif. |
| **Re** | Reynolds ρVc/μ. Petit Re = visqueux (maquettes). |
| **Mach** | V / son. < 0,3 ≈ incompressible. |
| **TSR λ** | Vitesse bout de pale / vitesse du vent. |
| **Station** | Coupe 2D de la pale à un rayon r (corde + vrillage). |
| **Vrillage** | Rotation de la section pour garder α ≈ α design. |
| **Fitness** | Note Darwin (selon le mode Score). |
| **Génération** | Un tour naissance → mutation → sélection. |
| **NACA MPXX** | Famille de profils : M cambrure, P position, XX épaisseur %. |

Utilisation libre et ouverte (licence MIT).
            """
        )


if __name__ == "__main__":
    main()
