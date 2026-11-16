"""
=============================================================================
PROJET PI² N°161 — Binôme 3
Module 2 : M-estimateur de Tyler (1987) et sa version régularisée
           (Chen, Wiesel & Hero, 2011 ; Wiesel 2012 ; Sun, Babu & Palomar 2014)
=============================================================================

Point fixe de Tyler (estimateur de DISPERSION / « scatter ») :

    S = (D/T) * sum_t  (x_t - mu)(x_t - mu)^T / [ (x_t - mu)^T S^{-1} (x_t - mu) ]

Version régularisée (shrinkage vers l'identité) :

    S <- (1 - lambda) * (D/T) * sum_t [...] + lambda * I_D

Normalisation de trace à CHAQUE itération : S <- D * S / Tr(S).
L'équation ne définit S qu'à une constante multiplicative près.

-----------------------------------------------------------------------------
PROPRIÉTÉS (cf. note de synthèse)
-----------------------------------------------------------------------------
* Invariance d'échelle par observation : x_t -> c_t x_t, c_t > 0 SCALAIRE.
  => neutralise les pics de volatilité COMMUNS (mélange d'échelles elliptique).
* Équivariance affine : S(A X) = A S(X) A^T (à une constante près).
  La régularisation vers I_D BRISE cette équivariance (arbitraire de la base).
* Existence / unicité (à échelle près) : T > D, positions générales, pas d'atome
  en 0. Si T <= D, il faut lambda > 0 (bornes explicites : Sun et al. 2014).
* Tyler est le MLE de la loi « Angular Central Gaussian » (ACG) :
      -log f(x ; S) = (1/2) log det S + (D/2) log(x^T S^{-1} x) + cte
  Ce critère est utilisé ici pour la sélection de lambda par validation croisée.

-----------------------------------------------------------------------------
RECALIBRAGE D'ÉCHELLE (obligatoire pour comparer à Sigma_true)
-----------------------------------------------------------------------------
    R_hat      = Delta^{-1/2} S Delta^{-1/2},  Delta = diag(S)
    sigma_i    = 1.4826 * MAD(X[:, i])
    Sigma_hat  = diag(sigma) R_hat diag(sigma)

ATTENTION : le facteur 1.4826 est calibré sur la gaussienne. Sous loi
leptokurtique (GARCH, Student), il SOUS-ESTIME l'écart-type. Les options
`scale_method="std"` / `"iqr"` / `"oracle"` permettent de quantifier ce biais.

-----------------------------------------------------------------------------
CONTRAT D'INTERFACE
-----------------------------------------------------------------------------
    estimate_tyler(X: np.ndarray, **kwargs) -> np.ndarray   # (D, D), SDP
=============================================================================
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.linalg import cholesky, solve_triangular

Array = np.ndarray

__all__ = [
    "estimate_tyler",
    "estimate_tyler_shape",
    "estimate_tyler_regularized",
    "tyler_fixed_point",
    "spatial_median",
    "robust_scale",
    "default_shrinkage",
    "select_shrinkage_cv",
]

_TINY = 1e-12


# =============================================================================
# Utilitaires numériques
# =============================================================================
def _symmetrize(A: Array) -> Array:
    return 0.5 * (A + A.T)


def _nearest_psd(A: Array, eps: float = 1e-12) -> Array:
    """Projette sur le cône SDP (filet de sécurité contre les erreurs d'arrondi)."""
    A = _symmetrize(A)
    w, V = np.linalg.eigh(A)
    floor = eps * max(float(w.max()), 1.0)
    if w.min() >= floor:
        return A
    return _symmetrize((V * np.maximum(w, floor)) @ V.T)


def _safe_cholesky(S: Array) -> Array:
    """
    Cholesky avec jitter croissant. En pratique, la régularisation lambda > 0
    et la normalisation de trace rendent l'échec très improbable.
    """
    S = _symmetrize(S)
    scale = max(float(np.trace(S)) / S.shape[0], _TINY)
    jitter = 0.0
    for k in range(8):
        try:
            return cholesky(S + jitter * np.eye(S.shape[0]), lower=True, check_finite=False)
        except Exception:
            jitter = 1e-12 * scale * (10.0**k)
    raise np.linalg.LinAlgError(
        "Factorisation de Cholesky impossible : matrice de dispersion dégénérée. "
        "Augmenter `shrinkage` (lambda > 0) ou vérifier les données."
    )


def _to_correlation(S: Array) -> Array:
    d = np.sqrt(np.maximum(np.diag(S), _TINY))
    R = S / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return _symmetrize(R)


# =============================================================================
# Centrage robuste
# =============================================================================
def spatial_median(X: Array, tol: float = 1e-9, max_iter: int = 500) -> Array:
    """
    Médiane spatiale (médiane géométrique) par l'algorithme de Weiszfeld :
        mu <- sum_t (x_t / ||x_t - mu||) / sum_t (1 / ||x_t - mu||)
    Plus efficace statistiquement que la médiane coordonnée par coordonnée,
    et équivariante par rotation (mais pas par mise à l'échelle diagonale).
    """
    mu = np.median(X, axis=0)
    for _ in range(max_iter):
        dist = np.linalg.norm(X - mu, axis=1)
        w = 1.0 / np.maximum(dist, 1e-12)  # gestion des points coïncidents
        mu_new = (X * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(mu_new - mu) < tol:
            return mu_new
        mu = mu_new
    return mu


def _center(X: Array, method: str, assume_centered: bool) -> Tuple[Array, Array]:
    """Retourne (donnees centrées, vecteur de centrage)."""
    if assume_centered or method == "none":
        return X.copy(), np.zeros(X.shape[1])
    if method == "median":
        mu = np.median(X, axis=0)
    elif method == "spatial":
        mu = spatial_median(X)
    elif method == "mean":
        mu = X.mean(axis=0)
    else:
        raise ValueError(f"Méthode de centrage inconnue : {method!r}")
    return X - mu, mu


# =============================================================================
# Échelle univariée robuste
# =============================================================================
def robust_scale(
    X: Array,
    method: str = "mad",
    Sigma_true: Optional[Array] = None,
) -> Array:
    """
    Estime un écart-type robuste pour chaque actif (vecteur de longueur D).

    "mad"    : 1.4826 * médiane(|X_i - médiane(X_i)|)   [défaut du cahier des charges]
               Calibration gaussienne : biais NÉGATIF sous queues lourdes.
    "iqr"    : (Q75 - Q25) / 1.349                      [calibration gaussienne aussi]
    "std"    : écart-type empirique (non robuste, mais quasi non biaisé pour la
               variance inconditionnelle ; utilisé comme ABLATION de diagnostic).
    "oracle" : sqrt(diag(Sigma_true)) — DIAGNOSTIC UNIQUEMENT, non implémentable
               en pratique ; isole l'erreur de forme de l'erreur d'échelle.
    """
    if method == "mad":
        med = np.median(X, axis=0)
        s = 1.4826 * np.median(np.abs(X - med), axis=0)
    elif method == "iqr":
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        s = (q75 - q25) / 1.349
    elif method == "std":
        s = X.std(axis=0, ddof=1)
    elif method == "oracle":
        if Sigma_true is None:
            raise ValueError("scale_method='oracle' requiert le mot-clé Sigma_true.")
        s = np.sqrt(np.diag(Sigma_true))
    else:
        raise ValueError(f"Méthode d'échelle inconnue : {method!r}")
    return np.maximum(s, 1e-10)  # évite une matrice singulière


# =============================================================================
# Cœur : itération de point fixe
# =============================================================================
def tyler_fixed_point(
    Xc: Array,
    shrinkage: float = 0.0,
    max_iter: int = 500,
    tol: float = 1e-6,
    init: Optional[Array] = None,
    return_info: bool = False,
) -> Union[Array, Tuple[Array, Dict[str, Any]]]:
    """
    Itération de Tyler sur données DÉJÀ centrées, avec normalisation de trace.

    Complexité par itération : O(T D^2 + D^3) (Cholesky + résolution triangulaire
    + produit matriciel), aucune inversion explicite.

    Critère d'arrêt : ||S_{k+1} - S_k||_F < tol (les deux matrices étant de trace D).
    """
    n, D = Xc.shape
    lam = float(shrinkage)
    if not (0.0 <= lam < 1.0):
        raise ValueError("Le paramètre de régularisation lambda doit être dans [0, 1[.")
    if lam == 0.0 and n <= D:
        raise ValueError(
            f"Tyler non régularisé requiert T > D (ici T={n}, D={D}). "
            "Utiliser shrinkage > 0 (cf. Chen–Wiesel–Hero 2011 / Sun et al. 2014)."
        )

    # Les observations nulles (x_t = mu) sont exclues : le point fixe n'y est pas défini.
    norms = np.linalg.norm(Xc, axis=1)
    keep = norms > 1e-12 * max(float(norms.max()), 1.0)
    if not np.all(keep):
        Xc = Xc[keep]
        n = Xc.shape[0]
        if n == 0:
            raise ValueError("Toutes les observations sont nulles après centrage.")
        if lam == 0.0 and n <= D:
            raise ValueError("T_effectif <= D après exclusion des observations nulles.")

    I_D = np.eye(D)
    S = I_D.copy() if init is None else _symmetrize(np.asarray(init, dtype=float))
    S *= D / np.trace(S)

    converged = False
    delta = np.inf
    Xt = Xc.T  # (D, n), réutilisé à chaque itération

    for k in range(1, max_iter + 1):
        L = _safe_cholesky(S)
        # Blanchiment : y_t = L^{-1} x_t  ==>  d_t = x_t^T S^{-1} x_t = ||y_t||^2
        Y = solve_triangular(L, Xt, lower=True, check_finite=False)
        d = np.einsum("ij,ij->j", Y, Y)
        d = np.maximum(d, _TINY)

        w = (D / n) / d
        S_new = (Xt * w) @ Xc  # (D, D)

        if lam > 0.0:
            S_new = (1.0 - lam) * S_new + lam * I_D

        S_new = _symmetrize(S_new)
        S_new *= D / np.trace(S_new)  # NORMALISATION DE TRACE (obligatoire)

        delta = float(np.linalg.norm(S_new - S, ord="fro"))
        S = S_new
        if delta < tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"Tyler : non-convergence après {max_iter} itérations "
            f"(||S_k+1 - S_k||_F = {delta:.3e} > tol = {tol:.1e}).",
            RuntimeWarning,
            stacklevel=2,
        )

    S = _nearest_psd(S)
    S *= D / np.trace(S)

    if return_info:
        return S, {"n_iter": k, "converged": converged, "delta": delta, "n_used": n}
    return S


# =============================================================================
# Sélection du paramètre de régularisation
# =============================================================================
def default_shrinkage(T: int, D: int) -> float:
    """
    Heuristique transparente et reproductible : lambda = (D/T) / (1 + D/T),
    bornée dans [0.02, 0.50]. Croît avec le ratio dimension/échantillon.

    Ce n'est PAS la règle de Chen–Wiesel–Hero (2011) ; c'est un choix par défaut
    documenté. Pour une sélection fondée sur les données, voir select_shrinkage_cv.
    """
    r = float(D) / float(max(T, 1))
    return float(np.clip(r / (1.0 + r), 0.02, 0.50))


def select_shrinkage_cv(
    X: Array,
    grid: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70),
    n_folds: int = 5,
    center: str = "median",
    max_iter: int = 200,
    tol: float = 1e-5,
    seed: Optional[int] = 0,
) -> float:
    """
    Sélection de lambda par validation croisée sur la vraisemblance ACG.

    Tyler étant le MLE de la loi Angular Central Gaussian, le critère
    hors échantillon naturel est :

        L(S) = log det(S) + D * mean_test log( x^T S^{-1} x ),   Tr(S) = D.

    Ce critère est invariant par x -> c x (à une constante près) : il sélectionne
    donc la FORME, indépendamment du recalibrage d'échelle. Coût : |grid| * n_folds
    ajustements ; à réserver aux expériences ponctuelles (pas aux 100 Monte-Carlo).
    """
    n, D = X.shape
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)

    scores = []
    for lam in grid:
        total, count = 0.0, 0
        for j in range(n_folds):
            te = folds[j]
            tr = np.concatenate([folds[m] for m in range(n_folds) if m != j])
            Xtr, _ = _center(X[tr], center, False)
            Xte, _ = _center(X[te], center, False)
            try:
                S = tyler_fixed_point(Xtr, shrinkage=lam, max_iter=max_iter, tol=tol)
                L = _safe_cholesky(S)
                Y = solve_triangular(L, Xte.T, lower=True, check_finite=False)
                d = np.maximum(np.einsum("ij,ij->j", Y, Y), _TINY)
                sign, logdet = np.linalg.slogdet(S)
                if sign <= 0:
                    raise np.linalg.LinAlgError
                total += float(logdet + D * np.mean(np.log(d))) * len(te)
                count += len(te)
            except Exception:
                total += 1e12 * len(te)
                count += len(te)
        scores.append(total / max(count, 1))

    return float(grid[int(np.argmin(scores))])


# =============================================================================
# Implémentation complète, typée
# =============================================================================
def _estimate_tyler_impl(
    X: Array,
    shrinkage: float = 0.0,
    rescale: bool = True,
    scale_method: str = "mad",
    center: str = "median",
    assume_centered: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
    auto_shrinkage: bool = False,
    cv_shrinkage: bool = False,
    init: Optional[Array] = None,
    Sigma_true: Optional[Array] = None,
    return_info: bool = False,
) -> Union[Array, Tuple[Array, Dict[str, Any]]]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X doit être de shape (T, D) ; reçu {X.shape}.")
    T, D = X.shape
    if not np.all(np.isfinite(X)):
        raise ValueError("X contient des valeurs non finies (NaN ou inf).")

    Xc, mu = _center(X, center, assume_centered)

    lam = float(shrinkage)
    if cv_shrinkage:
        lam = select_shrinkage_cv(X, center=center)
    elif auto_shrinkage:
        lam = default_shrinkage(T, D)

    S, info = tyler_fixed_point(
        Xc, shrinkage=lam, max_iter=max_iter, tol=tol, init=init, return_info=True
    )

    if rescale:
        sigma = robust_scale(X, method=scale_method, Sigma_true=Sigma_true)
        R = _to_correlation(S)
        Sigma_hat = _nearest_psd(R * np.outer(sigma, sigma))
    else:
        Sigma_hat = S  # scatter matrix normalisée : Tr = D

    info.update({"shrinkage": lam, "center": center, "mu": mu,
                 "rescale": rescale, "scale_method": scale_method if rescale else None})

    if return_info:
        return Sigma_hat, info
    return Sigma_hat


_ALLOWED_KWARGS = {
    "shrinkage", "lambda_", "rescale", "scale_method", "center", "assume_centered",
    "max_iter", "tol", "auto_shrinkage", "cv_shrinkage", "init", "Sigma_true",
    "return_info",
}


def estimate_tyler(X: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    ===================== CONTRAT D'INTERFACE COMMUN =====================
    estimate_tyler(X: np.ndarray, **kwargs) -> np.ndarray
        Entrée  : X de shape (T, D)
        Sortie  : Sigma_hat de shape (D, D), symétrique définie positive
    ======================================================================

    Mots-clés acceptés (tous optionnels)
    ------------------------------------
    shrinkage / lambda_ : float dans [0, 1[   (défaut 0.0)
    auto_shrinkage      : bool  -> lambda = default_shrinkage(T, D)
    cv_shrinkage        : bool  -> lambda choisi par validation croisée ACG
    rescale             : bool  (défaut True) -> recalibrage MAD de l'échelle
                          False -> retourne la scatter matrix de trace D
    scale_method        : "mad" (défaut) | "iqr" | "std" | "oracle"
    center              : "median" (défaut) | "spatial" | "mean" | "none"
    assume_centered     : bool (défaut False)
    max_iter            : int (défaut 500)
    tol                 : float (défaut 1e-6, critère de Frobenius)
    init                : (D, D) ou None
    Sigma_true          : requis uniquement si scale_method="oracle" (diagnostic)
    return_info         : bool -> retourne (Sigma_hat, dict d'informations)

    Les mots-clés inconnus sont IGNORÉS avec un avertissement, afin de tolérer
    les appels génériques des harness croisés entre binômes (ex. `delta=0.05`).
    """
    kw = dict(kwargs)
    if "lambda_" in kw:
        kw["shrinkage"] = kw.pop("lambda_")
    unknown = set(kw) - _ALLOWED_KWARGS
    if unknown:
        warnings.warn(
            f"estimate_tyler : mots-clés ignorés {sorted(unknown)}.", RuntimeWarning, stacklevel=2
        )
        for k in unknown:
            kw.pop(k)
    return _estimate_tyler_impl(X, **kw)


def estimate_tyler_shape(X: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Raccourci : scatter matrix de Tyler normalisée (Tr = D), sans recalibrage."""
    kwargs["rescale"] = False
    return estimate_tyler(X, **kwargs)


def estimate_tyler_regularized(X: np.ndarray, shrinkage: Optional[float] = None,
                               **kwargs: Any) -> np.ndarray:
    """Raccourci : Tyler régularisé ; lambda heuristique si non précisé."""
    if shrinkage is None:
        kwargs["auto_shrinkage"] = True
    else:
        kwargs["shrinkage"] = float(shrinkage)
    return estimate_tyler(X, **kwargs)