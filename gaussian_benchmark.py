"""Benchmark Monte-Carlo du régime calme gaussien.

Le script simule des rendements gaussiens pour plusieurs couples
dimension/taille d'échantillon, estime leur covariance avec différentes
méthodes, puis compare les estimations à la covariance théorique.

Les résultats comprennent des mesures d'erreur, le risque d'un portefeuille
de variance minimale et un graphique avec la moyenne et l'écart-type des
répétitions. Les fichiers sont enregistrés dans le dossier indiqué par
``--outdir``.

Exemples :
    python gaussian_benchmark.py --quick
    python gaussian_benchmark.py --n-mc 100 --seed 20261117
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gaussian_evaluation import evaluate_estimate, oracle_risk
from simulation_gaussian import simulate_gaussian

Array = np.ndarray
Estimator = Callable[[Array], Array]

CONFIGS = [
    {"name": "D10_T1000", "D": 10, "T": 1000},
    {"name": "D30_T300", "D": 30, "T": 300},
    {"name": "D50_T250", "D": 50, "T": 250},
    {"name": "D60_T120", "D": 60, "T": 120},
]
METRICS = ["frob_rel", "spectral", "corr_frob", "mvp_risk_ratio_c"]


def build_estimators() -> Dict[str, Estimator]:
    """Retourne les estimateurs de référence comparés dans ce benchmark.

    La covariance empirique sert de référence classique. Ledoit-Wolf applique
    une régularisation par shrinkage. D'autres estimateurs peuvent être ajoutés
    dans ``run_benchmark`` avec son paramètre ``estimators``.
    """
    return {
        "Empirique": lambda X: np.atleast_2d(np.cov(X, rowvar=False, ddof=1)),
        "Ledoit-Wolf": lambda X: LedoitWolf().fit(X).covariance_,
    }


def run_benchmark(
    n_mc: int = 100,
    configs: Optional[List[Dict[str, int]]] = None,
    base_seed: int = 20261117,
    w_max: Optional[float] = 0.20,
    estimators: Optional[Dict[str, Estimator]] = None,
) -> pd.DataFrame:
    """Exécute les simulations et retourne une ligne par estimateur et répétition.

    ``n_mc`` fixe le nombre de répétitions par configuration. ``configs`` décrit
    les dimensions et tailles d'échantillon, et ``base_seed`` rend les
    simulations reproductibles. ``w_max`` limite le poids d'un actif dans le
    portefeuille long-only. ``estimators`` permet de fournir d'autres fonctions
    qui prennent ``X`` en entrée et retournent une matrice de covariance.

    Chaque ligne contient les paramètres de simulation, les métriques calculées,
    la durée d'estimation et un message explicite si une estimation échoue.
    """
    if n_mc < 1:
        raise ValueError("n_mc doit être supérieur ou égal à 1.")
    configs = CONFIGS if configs is None else configs
    estimators = build_estimators() if estimators is None else estimators
    if not estimators:
        raise ValueError("Au moins un estimateur doit être fourni.")

    rows = []
    for config_index, config in enumerate(configs):
        dimension, sample_size = config["D"], config["T"]
        for replication in range(n_mc):
            seed = base_seed + config_index * 100_000 + replication
            simulation = simulate_gaussian(
                T=sample_size,
                D=dimension,
                seed=seed,
            )
            X = simulation["X"]
            sigma_true = simulation["Sigma_true"]
            reference = oracle_risk(sigma_true, w_max=w_max)

            for name, estimator in estimators.items():
                start = time.perf_counter()
                try:
                    estimate = estimator(X)
                    metrics = evaluate_estimate(
                        estimate,
                        sigma_true,
                        w_max=w_max,
                        ref_risk=reference,
                    )
                    error = ""
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    metrics = {metric: np.nan for metric in METRICS}
                    error = f"{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        "regime": "calme_gaussien",
                        "config": config["name"],
                        "D": dimension,
                        "T": sample_size,
                        "D_over_T": dimension / sample_size,
                        "estimator": name,
                        "replication": replication,
                        "seed": seed,
                        "time_s": time.perf_counter() - start,
                        "error": error,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Calcule la moyenne et l'écart-type des métriques par configuration."""
    grouping = ["regime", "config", "D_over_T", "estimator"]
    return results.groupby(grouping)[METRICS].agg(["mean", "std"])


def make_plots(results: pd.DataFrame, outdir: Path) -> Path:
    """Enregistre un graphique des métriques et retourne son chemin.

    Chaque panneau représente une métrique. Les points indiquent la moyenne des
    répétitions et les barres verticales leur écart-type, selon le ratio D/T.
    """
    titles = {
        "frob_rel": "Erreur de Frobenius relative",
        "spectral": "Erreur spectrale",
        "corr_frob": "Erreur de corrélation",
        "mvp_risk_ratio_c": "Risque MVP contraint / oracle",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(axes.ravel(), METRICS):
        for estimator, group in results.groupby("estimator"):
            summary = group.groupby("D_over_T")[metric].agg(["mean", "std"]).sort_index()
            axis.errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["std"].fillna(0.0),
                marker="o",
                capsize=3,
                label=estimator,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Ratio D/T (échelle logarithmique)")
        axis.set_title(titles[metric])
        axis.grid(alpha=0.3)
    axes[0, 0].legend()
    figure.suptitle("Régime calme gaussien — moyenne et écart-type Monte-Carlo")
    figure.tight_layout()
    path = outdir / "gaussian_benchmark.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    """Lit les options de commande et lance le benchmark complet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-mc",
        type=int,
        default=100,
        help="Nombre de répétitions par configuration (défaut : 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261117,
        help="Graine aléatoire pour reproduire les simulations.",
    )
    parser.add_argument(
        "--w-max",
        type=float,
        default=0.20,
        help="Poids maximal autorisé par actif dans le portefeuille.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results_gaussian"),
        help="Dossier où enregistrer les résultats.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Lance 3 répétitions par configuration pour un test rapide.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    results = run_benchmark(n_mc=3 if args.quick else args.n_mc, base_seed=args.seed, w_max=args.w_max)
    raw_path = args.outdir / "gaussian_benchmark_raw.csv"
    summary_path = args.outdir / "gaussian_benchmark_summary.csv"
    results.to_csv(raw_path, index=False)
    summary = summarize(results)
    summary.to_csv(summary_path)
    plot_path = make_plots(results, args.outdir)

    print(summary.to_string())
    print(f"\nRésultats détaillés : {raw_path}")
    print(f"Synthèse : {summary_path}")
    print(f"Graphique : {plot_path}")
    failures = results[results["error"] != ""]
    if not failures.empty:
        print(f"\nAttention : {len(failures)} estimation(s) ont échoué ; voir la colonne error.")


if __name__ == "__main__":
    main()
