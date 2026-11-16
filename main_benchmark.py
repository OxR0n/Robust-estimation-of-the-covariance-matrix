"""
=============================================================================
PROJET PI² N°161 — Binôme 3
Benchmark Monte-Carlo : 100 réplications
=============================================================================

Estimateurs comparés :
    1. Covariance empirique            (np.cov)
    2. Shrinkage Ledoit-Wolf           (sklearn.covariance.LedoitWolf)
    3. Tyler standard                  (lambda = 0)
    4. Tyler régularisé, échelle MAD   (lambda heuristique, défaut du projet)
    5. Tyler régularisé, échelle std   (ABLATION : isole le biais d'échelle du MAD)
    +  Portefeuille ORACLE (Sigma_true) comme référence de risque MVP.

Axes de variation :
    * persistance GARCH alpha + beta  in {0.85, 0.98}
    * ratio dimension / échantillon D/T in {0.01, 0.10, 0.20, 0.50}

Extensions (30 réplications) :
    queues lourdes (Student), choc structurel, régime elliptique exact,
    CCC hétérogène, haute dimension (T < D).

Exécution :
    python main_benchmark.py                 # 100 Monte-Carlo complets
    python main_benchmark.py --quick         # 5 Monte-Carlo (test de fumée)
    python main_benchmark.py --solver scipy  # plus rapide (recommandé en MC)
=============================================================================
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import matplotlib
matplotlib.use("Agg")  # exécution sans affichage (serveur / CI)
import matplotlib.pyplot as plt

from simulation_volatility import simulate_ccc_garch, simulate_scenario
from tyler_estimator import estimate_tyler, default_shrinkage
from evaluation import evaluate_estimate, oracle_risk

Estimator = Callable[[np.ndarray], np.ndarray]

# -----------------------------------------------------------------------------
# Grille principale : (D, T) balaye le ratio D/T
# -----------------------------------------------------------------------------
CONFIGS: List[Dict[str, Any]] = [
    {"name": "D10_T1000", "D": 10, "T": 1000},   # D/T = 0.01
    {"name": "D30_T300",  "D": 30, "T": 300},    # D/T = 0.10
    {"name": "D50_T250",  "D": 50, "T": 250},    # D/T = 0.20
    {"name": "D60_T120",  "D": 60, "T": 120},    # D/T = 0.50
]
PERSISTENCES = [0.85, 0.98]

METRICS = ["frob_rel", "spectral", "corr_frob", "mvp_risk_ratio_c", "mvp_risk_ratio"]


def build_estimators(D: int, T: int) -> Dict[str, Estimator]:
    """Ferme sur (D, T) pour calibrer automatiquement la régularisation."""
    lam = default_shrinkage(T, D)
    est: Dict[str, Estimator] = {
        "Empirique": lambda X: np.cov(X, rowvar=False),
        "Ledoit-Wolf": lambda X: LedoitWolf(assume_centered=False).fit(X).covariance_,
    }
    if T > D:
        est["Tyler (lambda=0)"] = lambda X: estimate_tyler(X, shrinkage=0.0, scale_method="mad")
    est["Tyler reg. (MAD)"] = lambda X: estimate_tyler(X, shrinkage=lam, scale_method="mad")
    est["Tyler reg. (std)*"] = lambda X: estimate_tyler(X, shrinkage=lam, scale_method="std")
    return est


def run_grid(
    n_mc: int,
    configs: List[Dict[str, Any]],
    persistences: List[float],
    base_seed: int,
    solver: str,
    w_max: Optional[float],
    scenario_kwargs: Optional[Dict[str, Any]] = None,
    scenario_name: Optional[str] = None,
    label: str = "principal",
    verbose: bool = True,
) -> pd.DataFrame:
    """Boucle Monte-Carlo générique : renvoie un DataFrame « tidy » (une ligne / run)."""
    rows: List[Dict[str, Any]] = []
    scenario_kwargs = scenario_kwargs or {}

    for cfg in configs:
        D, T = cfg["D"], cfg["T"]
        estimators = build_estimators(D, T)
        for pers in persistences:
            t0 = time.perf_counter()
            for m in range(n_mc):
                seed = base_seed + 1000 * m + 17 * D + T  # reproductible, sans recouvrement
                kw = dict(persistence=pers, alpha=0.12 if pers >= 0.95 else 0.08,
                          corr_structure="block", rho=0.6, rho_inter=0.15,
                          n_blocks=max(2, D // 10), uncond_vol=0.02,
                          store_Sigma_t=False, seed=seed)
                kw.update(scenario_kwargs)
                if scenario_name is not None:
                    kw.pop("persistence", None)
                    sim = simulate_scenario(scenario_name, T=T, D=D, **kw)
                else:
                    sim = simulate_ccc_garch(T=T, D=D, **kw)

                X, S_true = sim["X"], sim["Sigma_true"]
                ref = oracle_risk(S_true, w_max=w_max, solver=solver)

                for name, fn in estimators.items():
                    t1 = time.perf_counter()
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            S_hat = fn(X)
                        met = evaluate_estimate(S_hat, S_true, w_max=w_max,
                                                solver=solver, ref_risk=ref)
                    except Exception as exc:  # estimateur inapplicable (ex. T <= D)
                        met = {k: np.nan for k in
                               ["frob_rel", "spectral", "corr_frob",
                                "mvp_risk_ratio_c", "mvp_risk_ratio"]}
                        met["error"] = str(exc)[:80]
                    met.update({"label": label, "config": cfg["name"], "D": D, "T": T,
                                "D_over_T": D / T, "persistence": pers,
                                "estimator": name, "run": m,
                                "time_s": time.perf_counter() - t1})
                    rows.append(met)
            if verbose:
                print(f"  [{label}] {cfg['name']:>10s} | alpha+beta={pers:.2f} "
                      f"| {n_mc} runs en {time.perf_counter() - t0:.1f}s")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau « moyenne ± écart-type » demandé par le cahier des charges."""
    keys = ["label", "persistence", "config", "D_over_T", "estimator"]
    agg = df.groupby(keys)[METRICS].agg(["mean", "std"])
    cols = {}
    for met in METRICS:
        cols[met] = agg[(met, "mean")].map(lambda v: f"{v:8.4f}") + " ± " + \
                    agg[(met, "std")].map(lambda v: f"{v:7.4f}")
    out = pd.DataFrame(cols)
    out.index = out.index.set_names(keys)
    return out


def make_plots(df: pd.DataFrame, outdir: Path) -> None:
    """Quatre figures de synthèse pour le mémoire."""
    main = df[df["label"] == "principal"]
    if main.empty:
        return
    panel = [
        ("frob_rel", "Erreur de Frobenius relative"),
        ("spectral", "Erreur spectrale"),
        ("corr_frob", "Erreur de corrélation (Frobenius)"),
        ("mvp_risk_ratio_c", "Risque MVP contraint / risque oracle"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (met, title) in zip(axes.ravel(), panel):
        for pers in PERSISTENCES:
            sub = main[main["persistence"] == pers]
            for name, g in sub.groupby("estimator"):
                s = g.groupby("D_over_T")[met].mean().sort_index()
                ax.plot(s.index.values, s.values, marker="o",
                        label=f"{name} (p={pers:.2f})",
                        linestyle="--" if pers == 0.98 else "-")
        ax.set_xscale("log")
        ax.set_xlabel("Ratio D/T (échelle log)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("PI² 161 — Binôme 3 : Tyler vs références sous CCC-GARCH(1,1)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "benchmark_principal.png", dpi=160)
    plt.close(fig)

    # Figure « extensions » : effet du régime sur l'erreur de corrélation
    ext = df[df["label"] != "principal"]
    if not ext.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        piv = ext.pivot_table(index="label", columns="estimator", values="corr_frob", aggfunc="mean")
        piv.plot.bar(ax=ax, rot=20)
        ax.set_ylabel("Erreur de corrélation moyenne (Frobenius)")
        ax.set_title("Sensibilité au régime : queues lourdes, rupture, ellipticité, hétérogénéité")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(outdir / "benchmark_extensions.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Monte-Carlo Tyler / CCC-GARCH")
    parser.add_argument("--n-mc", type=int, default=100)
    parser.add_argument("--n-mc-ext", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="5 réplications (test rapide)")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--solver", choices=["auto", "cvxpy", "scipy"], default="scipy")
    parser.add_argument("--w-max", type=float, default=0.20,
                        help="Poids maximal du MVP contraint (None pour désactiver via -1)")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-ext", action="store_true")
    args = parser.parse_args()

    n_mc = 5 if args.quick else args.n_mc
    n_ext = 3 if args.quick else args.n_mc_ext
    w_max = None if args.w_max is not None and args.w_max < 0 else args.w_max
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PI² N°161 — Binôme 3 : régime à volatilité variable & M-estimateur de Tyler")
    print(f"Monte-Carlo : {n_mc} réplications | solveur MVP : {args.solver} | w_max = {w_max}")
    print("=" * 78)

    t0 = time.perf_counter()
    frames = [run_grid(n_mc, CONFIGS, PERSISTENCES, args.seed, args.solver, w_max,
                       label="principal")]

    if not args.no_ext:
        cfg_ext = [{"name": "D30_T300", "D": 30, "T": 300}]
        extensions = [
            ("queues_lourdes", "crise_student", {}),
            ("choc_structurel", "choc_structurel", {}),
            ("elliptique_exact", "elliptique_commun", {}),
            ("ccc_heterogene", "ccc_heterogene", {}),
        ]
        for label, name, kw in extensions:
            frames.append(run_grid(n_ext, cfg_ext, [0.98], args.seed + 777,
                                   args.solver, w_max, scenario_name=name,
                                   scenario_kwargs=kw, label=label))
        # Haute dimension : T < D => Tyler non régularisé inapplicable (démonstration)
        frames.append(run_grid(n_ext, [{"name": "D60_T40", "D": 60, "T": 40}], [0.98],
                               args.seed + 999, args.solver, w_max, label="T_inf_D"))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / "benchmark_brut.csv", index=False)

    table = summarize(df)
    table.to_csv(outdir / "benchmark_synthese.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 78)
    print("TABLEAU RÉCAPITULATIF — moyenne ± écart-type sur les réplications")
    print("(* = ablation : Tyler régularisé avec échelle empirique au lieu du MAD)")
    print("=" * 78)
    print(table.to_string())

    # Temps de calcul moyen par estimateur (argument d'implémentation pour le mémoire)
    print("\nTemps de calcul moyen par estimation (secondes) :")
    print(df.groupby("estimator")["time_s"].mean().sort_values().to_string())

    if not args.no_plots:
        make_plots(df, outdir)

    print(f"\nTerminé en {time.perf_counter() - t0:.1f}s. Résultats écrits dans « {outdir}/ ».")


if __name__ == "__main__":
    main()