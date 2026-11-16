"""
=============================================================================
PROJET PI² N°161 — Binôme 3
Module 3 : métriques d'erreur matricielle et portefeuille de variance minimale
=============================================================================

Normes d'erreur (vérité terrain Sigma_true connue) :
    * Frobenius absolue   : ||S_hat - S_true||_F
    * Frobenius relative  : ||S_hat - S_true||_F / ||S_true||_F
    * Spectrale (norme 2) : plus grande valeur singulière de (S_hat - S_true)
                            = max |valeur propre| pour une différence symétrique
    * Corrélation         : ||R_hat - R_true||_F   (structure pure, sans échelle)

Portefeuille de variance minimale (MVP) :
    * sans contrainte : w* = S_hat^{-1} 1 / (1^T S_hat^{-1} 1)
    * sous contraintes: min w^T S_hat w  s.c.  sum w = 1, 0 <= w <= w_max
    * VRAI risque hors échantillon : w*^T Sigma_true w*  (à comparer à l'oracle)
=============================================================================
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import minimize

Array = np.ndarray

try:  # cvxpy est optionnel : repli sur SLSQP si absent
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    _HAS_CVXPY = False

__all__ = [
    "to_correlation",
    "frobenius_error",
    "spectral_error",
    "correlation_error",
    "mvp_weights_analytic",
    "mvp_weights_constrained",
    "portfolio_variance",
    "evaluate_estimate",
    "oracle_risk",
]

_TINY = 1e-12


# =============================================================================
# Utilitaires
# =============================================================================
def _symmetrize(A: Array) -> Array:
    return 0.5 * (A + A.T)


def to_correlation(S: Array) -> Array:
    """Extrait la matrice de corrélation d'une matrice de covariance SDP."""
    d = np.sqrt(np.maximum(np.diag(S), _TINY))
    R = S / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return _symmetrize(R)


def _pseudo_inverse(S: Array, rcond: float = 1e-10) -> Array:
    """
    Pseudo-inverse symétrique tronquée : indispensable car la covariance
    empirique peut être quasi singulière (T proche de D), ce qui ferait
    exploser les poids MVP non contraints.
    """
    S = _symmetrize(S)
    w, V = np.linalg.eigh(S)
    thresh = rcond * max(float(w.max()), _TINY)
    inv_w = np.where(w > thresh, 1.0 / np.maximum(w, _TINY), 0.0)
    return _symmetrize((V * inv_w) @ V.T)


# =============================================================================
# Normes d'erreur
# =============================================================================
def frobenius_error(S_hat: Array, S_true: Array, relative: bool = False) -> float:
    E = np.asarray(S_hat) - np.asarray(S_true)
    err = float(np.linalg.norm(E, ord="fro"))
    if relative:
        denom = float(np.linalg.norm(S_true, ord="fro"))
        return err / max(denom, _TINY)
    return err


def spectral_error(S_hat: Array, S_true: Array, relative: bool = False) -> float:
    E = _symmetrize(np.asarray(S_hat) - np.asarray(S_true))
    err = float(np.abs(np.linalg.eigvalsh(E)).max())  # = norme 2 (différence symétrique)
    if relative:
        denom = float(np.abs(np.linalg.eigvalsh(_symmetrize(S_true))).max())
        return err / max(denom, _TINY)
    return err


def correlation_error(S_hat: Array, S_true: Array, relative: bool = False) -> float:
    """
    Erreur sur la STRUCTURE de corrélation uniquement. Métrique clé pour Tyler :
    elle neutralise totalement l'estimation d'échelle (invariance du M-estimateur).
    """
    E = to_correlation(S_hat) - to_correlation(S_true)
    err = float(np.linalg.norm(E, ord="fro"))
    if relative:
        denom = float(np.linalg.norm(to_correlation(S_true), ord="fro"))
        return err / max(denom, _TINY)
    return err


# =============================================================================
# Portefeuille de variance minimale
# =============================================================================
def mvp_weights_analytic(S: Array, rcond: float = 1e-10) -> Array:
    """w* = S^{-1} 1 / (1^T S^{-1} 1). Ventes à découvert autorisées."""
    S = np.asarray(S, dtype=float)
    ones = np.ones(S.shape[0])
    v = _pseudo_inverse(S, rcond=rcond) @ ones
    denom = float(ones @ v)
    if not np.isfinite(denom) or abs(denom) < 1e-14:
        warnings.warn("MVP analytique dégénéré : repli sur les poids 1/D.", RuntimeWarning)
        return ones / S.shape[0]
    return v / denom


def mvp_weights_constrained(
    S: Array,
    w_max: Optional[float] = None,
    long_only: bool = True,
    solver: str = "auto",
) -> Array:
    """
    MVP sous contraintes réalistes : sum w = 1, w >= 0 (long only), w <= w_max.

    solveur : "cvxpy" (ECOS/Clarabel), "scipy" (SLSQP avec gradient analytique),
              "auto" -> cvxpy si disponible, sinon scipy.
    Faisabilité : w_max >= 1/D est nécessaire.
    """
    S = _symmetrize(np.asarray(S, dtype=float))
    D = S.shape[0]
    if w_max is not None and w_max * D < 1.0 - 1e-12:
        raise ValueError(f"Contraintes infaisables : w_max={w_max} < 1/D={1.0 / D:.4f}.")

    use_cvxpy = _HAS_CVXPY if solver == "auto" else (solver == "cvxpy")
    if solver == "cvxpy" and not _HAS_CVXPY:
        raise ImportError("cvxpy n'est pas installé ; utiliser solver='scipy'.")

    if use_cvxpy:
        try:
            w = cp.Variable(D)
            S_par = cp.psd_wrap(S) if hasattr(cp, "psd_wrap") else S
            constraints = [cp.sum(w) == 1.0]
            if long_only:
                constraints.append(w >= 0.0)
            if w_max is not None:
                constraints.append(w <= w_max)
            prob = cp.Problem(cp.Minimize(cp.quad_form(w, S_par)), constraints)
            prob.solve()
            if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                out = np.asarray(w.value, dtype=float)
                out = np.clip(out, 0.0 if long_only else -np.inf, w_max if w_max else np.inf)
                return out / out.sum()
        except Exception:
            pass  # repli silencieux sur SLSQP

    # --- Repli : SLSQP avec gradient analytique ---
    upper = float(w_max) if w_max is not None else 1.0
    bounds = [(0.0 if long_only else -1.0, upper)] * D
    x0 = np.clip(mvp_weights_analytic(S), 0.0 if long_only else -1.0, upper)
    s = x0.sum()
    x0 = x0 / s if s > 1e-12 else np.full(D, 1.0 / D)

    res = minimize(
        fun=lambda w: (float(w @ S @ w), 2.0 * (S @ w)),
        x0=x0, jac=True, method="SLSQP", bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones(D)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = res.x if res.success else x0
    w = np.clip(w, 0.0 if long_only else -1.0, upper)
    return w / w.sum()


def portfolio_variance(w: Array, S: Array) -> float:
    """Variance du portefeuille : w^T S w (S = Sigma_true pour le VRAI risque)."""
    w = np.asarray(w, dtype=float)
    return float(w @ np.asarray(S, dtype=float) @ w)


def oracle_risk(Sigma_true: Array, w_max: Optional[float] = None,
                solver: str = "auto") -> Dict[str, float]:
    """Risque du portefeuille oracle (construit avec la vraie covariance)."""
    w_a = mvp_weights_analytic(Sigma_true)
    w_c = mvp_weights_constrained(Sigma_true, w_max=w_max, solver=solver)
    return {"mvp_var_oracle": portfolio_variance(w_a, Sigma_true),
            "mvp_var_oracle_c": portfolio_variance(w_c, Sigma_true)}


# =============================================================================
# Évaluation complète d'un estimateur
# =============================================================================
def evaluate_estimate(
    S_hat: Array,
    Sigma_true: Array,
    Sigma_t: Optional[Array] = None,
    w_max: Optional[float] = None,
    solver: str = "auto",
    ref_risk: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Calcule l'ensemble des métriques du rapport de cadrage.

    Clés retournées
    ---------------
    frob_abs, frob_rel, spectral, corr_frob, corr_rel,
    mvp_var, mvp_vol, mvp_risk_ratio            (MVP sans contrainte)
    mvp_var_c, mvp_vol_c, mvp_risk_ratio_c      (MVP sous contraintes)
    mvp_var_realized                            (si Sigma_t fourni : moyenne temporelle)
    cond, min_eig                               (diagnostics numériques)
    """
    S_hat = _symmetrize(np.asarray(S_hat, dtype=float))
    Sigma_true = _symmetrize(np.asarray(Sigma_true, dtype=float))

    if ref_risk is None:
        ref_risk = oracle_risk(Sigma_true, w_max=w_max, solver=solver)

    w_a = mvp_weights_analytic(S_hat)
    w_c = mvp_weights_constrained(S_hat, w_max=w_max, solver=solver)

    var_a = portfolio_variance(w_a, Sigma_true)
    var_c = portfolio_variance(w_c, Sigma_true)

    out: Dict[str, float] = {
        "frob_abs": frobenius_error(S_hat, Sigma_true),
        "frob_rel": frobenius_error(S_hat, Sigma_true, relative=True),
        "spectral": spectral_error(S_hat, Sigma_true),
        "corr_frob": correlation_error(S_hat, Sigma_true),
        "corr_rel": correlation_error(S_hat, Sigma_true, relative=True),
        "mvp_var": var_a,
        "mvp_vol": float(np.sqrt(max(var_a, 0.0))),
        "mvp_risk_ratio": var_a / max(ref_risk["mvp_var_oracle"], _TINY),
        "mvp_var_c": var_c,
        "mvp_vol_c": float(np.sqrt(max(var_c, 0.0))),
        "mvp_risk_ratio_c": var_c / max(ref_risk["mvp_var_oracle_c"], _TINY),
        "cond": float(np.linalg.cond(S_hat)),
        "min_eig": float(np.linalg.eigvalsh(S_hat).min()),
    }

    if Sigma_t is not None:
        # Risque réalisé exact, moyenné sur les covariances conditionnelles.
        out["mvp_var_realized"] = float(np.mean(np.einsum("i,tij,j->t", w_c, Sigma_t, w_c)))

    return out