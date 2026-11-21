"""Reproducible simulation of a calm, multivariate Gaussian market."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

Array = np.ndarray
Seed = Union[None, int, np.random.Generator]

__all__ = ["make_correlation_matrix", "simulate_gaussian"]


def _as_rng(seed: Seed) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def make_correlation_matrix(
    D: int,
    structure: str = "block",
    rho: float = 0.5,
    n_blocks: int = 3,
    rho_inter: float = 0.1,
    seed: Seed = None,
) -> Array:
    """Build a positive-definite correlation matrix for ``D`` assets."""
    if D < 1:
        raise ValueError("D doit être supérieur ou égal à 1.")

    idx = np.arange(D)
    if structure == "toeplitz":
        if not -1.0 < rho < 1.0:
            raise ValueError("Pour Toeplitz, rho doit être dans ]-1, 1[.")
        R = rho ** np.abs(idx[:, None] - idx[None, :])
    elif structure == "equicorrelation":
        if D > 1 and not -1.0 / (D - 1) < rho < 1.0:
            raise ValueError("rho ne définit pas une matrice équicorrélée positive.")
        R = np.full((D, D), rho, dtype=float)
        np.fill_diagonal(R, 1.0)
    elif structure == "block":
        if not -1.0 < rho < 1.0 or not -1.0 < rho_inter < 1.0:
            raise ValueError("rho et rho_inter doivent être dans ]-1, 1[.")
        labels = np.empty(D, dtype=int)
        for block, members in enumerate(np.array_split(idx, max(1, min(n_blocks, D)))):
            labels[members] = block
        R = np.where(labels[:, None] == labels[None, :], rho, rho_inter).astype(float)
        np.fill_diagonal(R, 1.0)
    elif structure == "random":
        rng = _as_rng(seed)
        factors = rng.standard_normal((D, max(1, D // 3)))
        covariance = factors @ factors.T + 0.5 * np.eye(D)
        scale = np.sqrt(np.diag(covariance))
        R = covariance / np.outer(scale, scale)
    else:
        raise ValueError(f"Structure de corrélation inconnue : {structure!r}")

    eigenvalues = np.linalg.eigvalsh(R)
    if eigenvalues.min() <= 1e-12:
        raise ValueError(
            f"La structure {structure!r} ne produit pas une matrice "
            "de corrélation définie positive."
        )
    return 0.5 * (R + R.T)


def simulate_gaussian(
    T: int,
    D: int,
    mean: Union[float, Array] = 0.0,
    volatility: Union[float, Array] = 0.02,
    corr_structure: str = "block",
    rho: float = 0.5,
    n_blocks: int = 3,
    rho_inter: float = 0.1,
    seed: Seed = None,
    output: str = "dict",
) -> Union[Dict[str, Any], Tuple[Array, Array]]:
    """Simulate iid returns ``X ~ N(mean, Sigma_true)``.

    The returned dictionary follows the shared regime interface: ``X`` has
    shape ``(T, D)`` and ``Sigma_true`` has shape ``(D, D)``.
    """
    if T < 2:
        raise ValueError("T doit être supérieur ou égal à 2.")
    if D < 1:
        raise ValueError("D doit être supérieur ou égal à 1.")
    if output not in ("dict", "tuple"):
        raise ValueError("output doit valoir 'dict' ou 'tuple'.")

    rng = _as_rng(seed)
    mu = np.broadcast_to(np.asarray(mean, dtype=float), (D,)).copy()
    sigma = np.broadcast_to(np.asarray(volatility, dtype=float), (D,)).copy()
    if not np.all(np.isfinite(mu)):
        raise ValueError("mean doit contenir uniquement des valeurs finies.")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Les volatilités doivent être finies et strictement positives.")

    correlation = make_correlation_matrix(
        D=D,
        structure=corr_structure,
        rho=rho,
        n_blocks=n_blocks,
        rho_inter=rho_inter,
        seed=rng,
    )
    sigma_true = correlation * np.outer(sigma, sigma)
    X = rng.multivariate_normal(mu, sigma_true, size=T, check_valid="raise")

    if output == "tuple":
        return X, sigma_true
    return {
        "X": X,
        "Sigma_true": sigma_true,
        "R_true": correlation,
        "params": {
            "T": T,
            "D": D,
            "mean": mu,
            "volatility": sigma,
            "corr_structure": corr_structure,
            "rho": rho,
            "n_blocks": n_blocks,
            "rho_inter": rho_inter,
            "seed": seed if isinstance(seed, (int, type(None))) else None,
            "distribution": "multivariate_normal",
        },
    }
