from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from darwin_hor.config import EvoConfig
from darwin_hor.geometry import (
    Individual,
    apply_flap,
    apply_le_radius,
    apply_te_gap,
    as_kulfan,
    geometry_is_sane,
    interpolate_airfoils,
    mutate_cst,
    random_naca4,
)
from darwin_hor.solver import evaluate_individual

MUTATIONS = ("inter", "tegap", "lerad", "flap", "cst")


@dataclass
class GenerationStats:
    generation: int
    n_alive: int
    best_fitness: float
    median_fitness: float
    best_name: str
    best_thickness: float
    best_cl_cd_mean: float
    population: list[Individual] = field(repr=False)


@dataclass
class EvolutionResult:
    best: Individual
    history: list[GenerationStats]
    population: list[Individual]
    config: EvoConfig
    run_id: int


def _new_naca(rng: np.random.Generator, gen: int, uid: int, cfg: EvoConfig) -> Individual:
    af = random_naca4(
        rng,
        n_points=cfg.n_points,
        thickness_min=cfg.thickness_min,
        thickness_max=cfg.thickness_max,
    )
    k = as_kulfan(af, name=af.name)
    return Individual(
        airfoil=k,
        name=k.name,
        generation=gen,
        mutation="naca",
        parents=(),
        uid=uid,
    )


def _mutate_one(
    parent: Individual,
    population: list[Individual],
    rng: np.random.Generator,
    gen: int,
    uid: int,
    cfg: EvoConfig,
) -> Optional[Individual]:
    op = MUTATIONS[int(rng.integers(0, len(MUTATIONS)))]
    name = f"g{gen}_{uid:04d}_{op}"
    try:
        if op == "inter":
            other = population[int(rng.integers(0, len(population)))]
            frac = float(rng.uniform(0.4, 0.6))
            child_af = interpolate_airfoils(parent.airfoil, other.airfoil, frac, name)
            parents = (parent.name, other.name)
        elif op == "tegap":
            gap = float(rng.random() / 50.0)
            child_af = apply_te_gap(parent.airfoil, gap, name)
            parents = (parent.name,)
        elif op == "lerad":
            scale = float(rng.random() + 0.5)
            child_af = apply_le_radius(parent.airfoil, scale, name)
            parents = (parent.name,)
        elif op == "flap":
            hinge = float(rng.uniform(0.2, 0.8))
            deflection = float(rng.uniform(-3.0, 3.0))
            child_af = apply_flap(parent.airfoil, hinge, deflection, name)
            parents = (parent.name,)
        else:
            child_af = mutate_cst(parent.airfoil, rng, sigma=0.035, name=name)
            parents = (parent.name,)
    except Exception:
        return None

    if not geometry_is_sane(child_af, cfg.thickness_min, cfg.thickness_max):
        return None
    return Individual(
        airfoil=child_af,
        name=name,
        generation=gen,
        mutation=op,
        parents=parents,
        uid=uid,
    )


def _select(
    population: list[Individual],
    rng: np.random.Generator,
    cfg: EvoConfig,
) -> list[Individual]:
    alive = [p for p in population if p.fitness > 0]
    if not alive:
        return population[: cfg.pop_size]
    alive.sort(key=lambda p: p.fitness, reverse=True)
    elites = alive[: max(1, cfg.n_elites)]
    elite_names = {e.name for e in elites}
    fitnesses = np.array([p.fitness for p in alive], dtype=float)
    median = float(np.median(fitnesses))

    kept = list(elites)
    for ind in alive:
        if ind.name in elite_names:
            continue
        if ind.fitness >= median:
            if rng.random() >= cfg.elite_death:
                kept.append(ind)
        elif rng.random() < cfg.weak_survival:
            kept.append(ind)

    cap = max(cfg.pop_size, min(len(kept), cfg.pop_size * 2))
    kept.sort(key=lambda p: p.fitness, reverse=True)
    return kept[:cap]


def run_evolution(
    cfg: EvoConfig,
    on_generation: Optional[Callable[[GenerationStats], None]] = None,
) -> EvolutionResult:
    rng = np.random.default_rng(cfg.seed)
    run_id = int(rng.integers(100, 1000))
    uid = 0
    population: list[Individual] = []
    max_tries = max(80, cfg.pop_size * 25)
    while len(population) < cfg.pop_size and uid < max_tries:
        uid += 1
        ind = _new_naca(rng, gen=0, uid=uid, cfg=cfg)
        if not geometry_is_sane(ind.airfoil, cfg.thickness_min, cfg.thickness_max):
            continue
        evaluate_individual(ind, cfg)
        population.append(ind)
    if not population:
        raise RuntimeError(
            f"Aucun NACA n'entre dans [{100 * cfg.thickness_min:.1f} %, "
            f"{100 * cfg.thickness_max:.1f} %] d'épaisseur. "
            "Élargis « Épaisseur min / max » : un NACA4412 fait 12 %, un NACA0021 fait 21 %. "
            "Si le max est à 8 %, tous les profils de départ (10–21 % dans la V2) étaient refusés."
        )

    history: list[GenerationStats] = []

    def snapshot(gen: int) -> GenerationStats:
        ranked = sorted(population, key=lambda p: p.fitness, reverse=True)
        fits = np.array([p.fitness for p in ranked], dtype=float)
        best = ranked[0]
        stats = GenerationStats(
            generation=gen,
            n_alive=len(ranked),
            best_fitness=float(best.fitness),
            median_fitness=float(np.median(fits)),
            best_name=best.name,
            best_thickness=float(best.thickness),
            best_cl_cd_mean=float(best.cl_cd_mean),
            population=list(ranked),
        )
        history.append(stats)
        if on_generation is not None:
            on_generation(stats)
        return stats

    snapshot(0)

    for gen in range(1, cfg.n_generations + 1):
        children: list[Individual] = []
        parents = list(population)
        for parent in parents:
            uid += 1
            child = _mutate_one(parent, parents, rng, gen, uid, cfg)
            if child is not None:
                children.append(child)

        for _ in range(cfg.naca_injection):
            uid += 1
            children.append(_new_naca(rng, gen=gen, uid=uid, cfg=cfg))

        for child in children:
            evaluate_individual(child, cfg)

        merged = parents + children
        merged = [
            p
            for p in merged
            if p.fitness > 0 and geometry_is_sane(p.airfoil, cfg.thickness_min * 0.85, cfg.thickness_max * 1.05)
        ]
        population = _select(merged, rng, cfg)
        if len(population) < max(4, cfg.pop_size // 2):
            for _ in range(cfg.pop_size - len(population)):
                uid += 1
                extra = _new_naca(rng, gen=gen, uid=uid, cfg=cfg)
                evaluate_individual(extra, cfg)
                population.append(extra)
        snapshot(gen)

    best = max(population, key=lambda p: p.fitness)
    return EvolutionResult(
        best=best,
        history=history,
        population=sorted(population, key=lambda p: p.fitness, reverse=True),
        config=cfg,
        run_id=run_id,
    )
