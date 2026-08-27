from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darwin-hor", description="Darwin HOR 3")
    parser.add_argument("--cli", action="store_true", help="Lancer une évolution en console (sans Streamlit)")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--pop", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.cli:
        app = Path(__file__).resolve().parent.parent / "app.py"
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", str(app), "--browser.gatherUsageStats=false"]
        stcli.main()
        return 0

    from darwin_hor.config import EvoConfig
    from darwin_hor.evolution import run_evolution

    cfg = EvoConfig(n_generations=args.generations, pop_size=args.pop, seed=args.seed)

    def _print(stats):
        print(
            f"gen {stats.generation:3d}  n={stats.n_alive:3d}  "
            f"best={stats.best_fitness:7.2f}  median={stats.median_fitness:7.2f}  "
            f"{stats.best_name}"
        )

    result = run_evolution(cfg, on_generation=_print)
    print(f"Champion: {result.best.name}  fitness={result.best.fitness:.2f}  Cl/Cd={result.best.cl_cd_mean:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
