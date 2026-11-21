"""Common covariance and minimum-variance portfolio evaluation metrics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize

Array = np.ndarray
_TINY = 1e-12

__all__ = [
    "correlation_error",
    "evaluate_estimate",
    "frobenius_error",
    "mvp_weights_analytic",
    "mvp_weights_constrained",
    "oracle_risk",
    "portfolio_variance",
    "spectral_error",
    "to_correlation",
]


def _symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)


def _validated_covariance(matrix: Array, name: str) -> Array:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} doit être une matrice carrée.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contient des valeurs non finies.")
    matrix = _symmetrize(matrix)
    if np.linalg.eigvalsh(matrix).min() < -1e-10:
        raise ValueError(f"{name} doit être positive semi-définie.")
    return matrix


def to_correlation(covariance: Array) -> Array:
    covariance = _validated_covariance(covariance, "covariance")
    diagonal = np.sqrt(np.maximum(np.diag(covariance), _TINY))
    result = covariance / np.outer(diagonal, diagonal)
    np.fill_diagonal(result, 1.0)
    return _symmetrize(result)


def frobenius_error(estimate: Array, truth: Array, relative: bool = False) -> float:
    error = float(np.linalg.norm(np.asarray(estimate) - np.asarray(truth), ord="fro"))
    if not relative:
        return error
    return error / max(float(np.linalg.norm(truth, ord="fro")), _TINY)


def spectral_error(estimate: Array, truth: Array, relative: bool = False) -> float:
    difference = _symmetrize(np.asarray(estimate) - np.asarray(truth))
    error = float(np.max(np.abs(np.linalg.eigvalsh(difference))))
    if not relative:
        return error
    scale = float(np.max(np.abs(np.linalg.eigvalsh(_symmetrize(truth)))))
    return error / max(scale, _TINY)


def correlation_error(estimate: Array, truth: Array, relative: bool = False) -> float:
    error = float(np.linalg.norm(to_correlation(estimate) - to_correlation(truth), ord="fro"))
    if not relative:
        return error
    scale = float(np.linalg.norm(to_correlation(truth), ord="fro"))
    return error / max(scale, _TINY)


def mvp_weights_analytic(covariance: Array) -> Array:
    """Unconstrained minimum-variance weights, allowing short positions."""
    covariance = _validated_covariance(covariance, "covariance")
    ones = np.ones(covariance.shape[0])
    weights = np.linalg.pinv(covariance, hermitian=True) @ ones
    denominator = float(ones @ weights)
    if not np.isfinite(denominator) or abs(denominator) < _TINY:
        raise np.linalg.LinAlgError("Impossible de calculer les poids MVP analytiques.")
    return weights / denominator


def mvp_weights_constrained(
    covariance: Array,
    w_max: Optional[float] = None,
    long_only: bool = True,
) -> Array:
    """Solve the minimum-variance problem with common portfolio constraints."""
    covariance = _validated_covariance(covariance, "covariance")
    dimension = covariance.shape[0]
    if w_max is not None and (w_max <= 0 or w_max * dimension < 1.0 - 1e-12):
        raise ValueError("w_max doit être positif et au moins égal à 1 / D.")

    bounds = [
        (0.0 if long_only else None, w_max if w_max is not None else (1.0 if long_only else None))
        for _ in range(dimension)
    ]
    initial = np.full(dimension, 1.0 / dimension)
    scale = max(float(np.max(np.abs(covariance))), _TINY)
    scaled = covariance / scale
    result = minimize(
        fun=lambda weights: float(weights @ scaled @ weights),
        x0=initial,
        jac=lambda weights: 2.0 * scaled @ weights,
        method="SLSQP",
        bounds=bounds,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
            "jac": lambda weights: np.ones(dimension),
        },
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"L'optimisation MVP a échoué : {result.message}")
    return np.asarray(result.x, dtype=float)


def portfolio_variance(weights: Array, covariance: Array) -> float:
    return float(weights @ covariance @ weights)


def oracle_risk(
    sigma_true: Array,
    w_max: Optional[float] = None,
) -> Dict[str, float]:
    sigma_true = _validated_covariance(sigma_true, "Sigma_true")
    weights_analytic = mvp_weights_analytic(sigma_true)
    weights_constrained = mvp_weights_constrained(sigma_true, w_max=w_max)
    return {
        "mvp_var_oracle": portfolio_variance(weights_analytic, sigma_true),
        "mvp_var_oracle_c": portfolio_variance(weights_constrained, sigma_true),
    }


def evaluate_estimate(
    estimate: Array,
    sigma_true: Array,
    w_max: Optional[float] = None,
    ref_risk: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Evaluate matrix error and out-of-sample risk against the true covariance."""
    estimate = _validated_covariance(estimate, "Sigma_hat")
    sigma_true = _validated_covariance(sigma_true, "Sigma_true")
    if estimate.shape != sigma_true.shape:
        raise ValueError("Sigma_hat et Sigma_true doivent avoir la même dimension.")

    if ref_risk is None:
        ref_risk = oracle_risk(sigma_true, w_max=w_max)
    weights_analytic = mvp_weights_analytic(estimate)
    weights_constrained = mvp_weights_constrained(estimate, w_max=w_max)
    risk_analytic = portfolio_variance(weights_analytic, sigma_true)
    risk_constrained = portfolio_variance(weights_constrained, sigma_true)
    return {
        "frob_abs": frobenius_error(estimate, sigma_true),
        "frob_rel": frobenius_error(estimate, sigma_true, relative=True),
        "spectral": spectral_error(estimate, sigma_true),
        "corr_frob": correlation_error(estimate, sigma_true),
        "corr_rel": correlation_error(estimate, sigma_true, relative=True),
        "mvp_var": risk_analytic,
        "mvp_vol": float(np.sqrt(max(risk_analytic, 0.0))),
        "mvp_risk_ratio": risk_analytic / max(ref_risk["mvp_var_oracle"], _TINY),
        "mvp_var_c": risk_constrained,
        "mvp_vol_c": float(np.sqrt(max(risk_constrained, 0.0))),
        "mvp_risk_ratio_c": risk_constrained / max(ref_risk["mvp_var_oracle_c"], _TINY),
        "min_eig": float(np.linalg.eigvalsh(estimate).min()),
        "cond": float(np.linalg.cond(estimate)),
    }
