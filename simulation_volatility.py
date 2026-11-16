"""
=============================================================================
PROJET PI² N°161 — Binôme 3
Module 1 : simulation d'un marché à volatilité variable (hétéroscédasticité)
=============================================================================

Modèle principal : CCC-GARCH(1,1) multivarié (Constant Conditional Correlation)

    sigma_{i,t}^2 = omega_i + alpha_i * x_{i,t-1}^2 + beta_i * sigma_{i,t-1}^2
    x_t           = D_t z_t,   D_t = diag(sigma_{1,t}, ..., sigma_{D,t})
    z_t           ~ N(0, R)  i.i.d.   (ou Student-t multivariée normalisée)
    Sigma_t       = D_t R D_t          (covariance conditionnelle)

Vérité terrain (covariance inconditionnelle, analytique) :

    bar_sigma_i^2 = omega_i / (1 - alpha_i - beta_i)
    Sigma_true    = bar_D R bar_D

Paramétrisation inverse : omega_i = bar_sigma_i^2 * (1 - alpha_i - beta_i),
ce qui découple la covariance cible de la persistance GARCH. Indispensable
pour comparer les régimes alpha+beta = 0.85 et 0.98 à Sigma_true identique.

-----------------------------------------------------------------------------
AVERTISSEMENT THÉORIQUE (cf. note de synthèse, §1.3-A)
-----------------------------------------------------------------------------
Le M-estimateur de Tyler est invariant par x_t -> c_t x_t avec c_t SCALAIRE.
Il est donc exactement justifié sous un modèle à volatilité COMMUNE
(Sigma_t = h_t * Sigma, loi elliptique), et seulement approximativement sous
CCC-GARCH à dynamiques hétérogènes. L'option `vol_model="common"` fournit ce
régime de validation exacte :

    h_t = omega_h + alpha_h * h_{t-1} * e_{t-1}^2 + beta_h * h_{t-1},
    e_t ~ N(0,1) i.i.d. et INDÉPENDANT de z_t,
    x_t = sqrt(h_t) * bar_D z_t   ==>   Sigma_t = h_t * Sigma_true.

-----------------------------------------------------------------------------
CONTRAT D'INTERFACE (commun aux 3 binômes)
-----------------------------------------------------------------------------
Toute fonction de simulation retourne, au choix :
  * output="dict"  : dictionnaire contenant au minimum "X" et "Sigma_true"
  * output="tuple" : le tuple (X, Sigma_true)
avec X de shape (T, D) et Sigma_true de shape (D, D), symétrique définie positive.
=============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

Array = np.ndarray

__all__ = [
    "make_correlation_matrix",
    "make_garch_params",
    "simulate_ccc_garch",
    "simulate_scenario",
    "SCENARIOS",
]


# =============================================================================
# Utilitaires d'algèbre linéaire (volontairement auto-suffisants)
# =============================================================================
def _as_rng(seed: Union[None, int, np.random.Generator]) -> np.random.Generator:
    """Normalise l'argument de graine en un objet Generator (reproductibilité)."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _nearest_correlation_matrix(R: Array, eps: float = 1e-10, max_iter: int = 100) -> Array:
    """
    Projette une matrice quasi-corrélation sur l'ensemble des matrices de
    corrélation définies positives (alternance : clipping spectral / normalisation
    de la diagonale). Simple filet de sécurité numérique : toutes les structures
    proposées ci-dessous sont PD par construction.
    """
    R = 0.5 * (R + R.T)
    for _ in range(max_iter):
        w, V = np.linalg.eigh(R)
        if w.min() > eps:
            break
        R = (V * np.maximum(w, eps)) @ V.T
        d = np.sqrt(np.maximum(np.diag(R), eps))
        R = R / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return 0.5 * (R + R.T)


# =============================================================================
# Structures de corrélation
# =============================================================================
def make_correlation_matrix(
    D: int,
    structure: str = "toeplitz",
    rho: float = 0.5,
    n_blocks: int = 3,
    rho_inter: float = 0.1,
    seed: Union[None, int, np.random.Generator] = None,
) -> Array:
    """
    Construit une matrice de corrélation R (D x D) valide (symétrique, PD, diag = 1).

    Paramètres
    ----------
    D : nombre d'actifs.
    structure :
        - "toeplitz"        : R_ij = rho^|i-j|  (décroissance avec la « distance »)
        - "block"           : secteurs financiers contigus, corrélation intra = rho,
                              inter-secteurs = rho_inter
        - "equicorrelation" : R_ij = rho pour i != j
        - "random"          : structure factorielle aléatoire (chargements gaussiens)
    rho : corrélation intra (ou paramètre de Toeplitz), |rho| < 1.
    n_blocks : nombre de secteurs pour la structure "block".
    rho_inter : corrélation inter-secteurs (>= 0 recommandé).
    """
    rng = _as_rng(seed)
    idx = np.arange(D)

    if structure == "toeplitz":
        R = np.abs(rho) ** np.abs(idx[:, None] - idx[None, :]).astype(float)

    elif structure == "block":
        # Secteurs CONTIGUS : plus réaliste qu'un découpage modulo pour des actifs
        # triés par industrie.
        edges = np.array_split(idx, max(1, min(n_blocks, D)))
        labels = np.empty(D, dtype=int)
        for b, e in enumerate(edges):
            labels[e] = b
        same_block = labels[:, None] == labels[None, :]
        R = np.where(same_block, rho, rho_inter).astype(float)
        np.fill_diagonal(R, 1.0)

    elif structure == "equicorrelation":
        R = np.full((D, D), float(rho))
        np.fill_diagonal(R, 1.0)

    elif structure == "random":
        k = max(2, D // 3)
        B = rng.standard_normal((D, k))
        C = B @ B.T + 0.5 * np.eye(D)  # le terme diagonal garantit la PD
        d = np.sqrt(np.diag(C))
        R = C / np.outer(d, d)

    else:
        raise ValueError(f"Structure de corrélation inconnue : {structure!r}")

    return _nearest_correlation_matrix(R)


# =============================================================================
# Paramètres GARCH
# =============================================================================
def make_garch_params(
    D: int,
    persistence: float = 0.95,
    alpha: float = 0.10,
    uncond_vol: Union[float, Array] = 0.02,
    heterogeneous: bool = False,
    seed: Union[None, int, np.random.Generator] = None,
) -> Dict[str, Array]:
    """
    Construit (omega, alpha, beta) par actif avec contraintes de stationnarité.

    persistence : cible alpha + beta (< 1, stationnarité au second ordre).
    alpha       : coefficient de réaction aux chocs (ARCH).
    uncond_vol  : écart-type inconditionnel bar_sigma_i (scalaire ou vecteur (D,)).
                  0.02 = 2 % par jour, soit ~31.7 % annualisé (sqrt(252)).
    heterogeneous : si True, tire alpha_i et (alpha+beta)_i par actif autour des
                  valeurs cibles. ATTENTION : cela brise l'ellipticité exacte et
                  dégrade la cible population de Tyler (cf. §1.3-A).
    """
    rng = _as_rng(seed)

    if not (0.0 <= persistence < 1.0):
        raise ValueError("La persistance alpha+beta doit vérifier 0 <= alpha+beta < 1.")

    p = np.full(D, float(persistence))
    a = np.full(D, float(alpha))

    if heterogeneous:
        p = np.clip(p + rng.uniform(-0.03, 0.03, D), 0.0, 0.995)
        a = np.clip(a * rng.uniform(0.5, 1.5, D), 1e-4, None)

    a = np.clip(a, 0.0, np.maximum(p - 1e-6, 0.0))  # garantit beta >= 0
    b = p - a

    v = np.broadcast_to(np.asarray(uncond_vol, dtype=float), (D,)).astype(float).copy()
    if np.any(v <= 0):
        raise ValueError("Les volatilités inconditionnelles doivent être > 0.")

    omega = v**2 * (1.0 - p)  # paramétrisation INVERSE : bar_sigma^2 imposée

    assert np.all(omega > 0) and np.all(a >= 0) and np.all(b >= 0) and np.all(a + b < 1)

    return {"omega": omega, "alpha": a, "beta": b, "uncond_var": v**2, "uncond_vol": v}


# =============================================================================
# Simulateur principal
# =============================================================================
def simulate_ccc_garch(
    T: int,
    D: int,
    persistence: float = 0.95,
    alpha: float = 0.10,
    uncond_vol: Union[float, Array] = 0.02,
    corr_structure: str = "toeplitz",
    rho: float = 0.5,
    n_blocks: int = 3,
    rho_inter: float = 0.1,
    innovation: str = "gaussian",
    df: float = 5.0,
    vol_model: str = "ccc",
    heterogeneous_dynamics: bool = False,
    break_at: Optional[float] = None,
    persistence_after: Optional[float] = None,
    alpha_after: Optional[float] = None,
    vol_multiplier_after: float = 2.0,
    rho_after: Optional[float] = None,
    burn: int = 500,
    seed: Union[None, int, np.random.Generator] = None,
    store_Sigma_t: bool = True,
    output: str = "dict",
) -> Union[Dict[str, Any], Tuple[Array, Array]]:
    """
    Simule T observations d'un marché à volatilité variable, avec vérité terrain.

    Paramètres clés
    ---------------
    vol_model : "ccc"    -> CCC-GARCH par actif (spécification du cahier des charges)
                "common" -> facteur de volatilité SCALAIRE commun (loi elliptique :
                            régime de validation théorique exacte de Tyler)
    innovation : "gaussian" | "student" (Student-t multivariée NORMALISÉE pour que
                 Cov(z_t) = R, donc Sigma_true inchangée ; requiert df > 2).
    break_at : fraction de l'échantillon où survient un choc structurel (ex. 0.5).
               Les paramètres GARCH, la volatilité cible et la corrélation basculent.
    store_Sigma_t : si True, retourne l'historique (T, D, D) des covariances
                  conditionnelles (utile pour le risque réalisé exact).
    output : "dict" (défaut) ou "tuple" -> (X, Sigma_true).

    Retour (dict)
    -------------
    "X"                 : (T, D) rendements simulés
    "Sigma_true"        : (D, D) covariance inconditionnelle analytique (cible)
    "Sigma_t"           : (T, D, D) covariances conditionnelles ou None
    "R_true"            : corrélation associée à Sigma_true
    "Sigma_true_regimes": (Sigma_1, Sigma_2) cibles des deux régimes
    "cond_var"          : (T, D) variances conditionnelles
    "params"            : métadonnées complètes (reproductibilité)
    """
    rng = _as_rng(seed)
    T, D = int(T), int(D)
    if T < 2 or D < 1:
        raise ValueError("Il faut T >= 2 et D >= 1.")
    if vol_model not in ("ccc", "common"):
        raise ValueError("vol_model doit valoir 'ccc' ou 'common'.")
    if innovation not in ("gaussian", "student"):
        raise ValueError("innovation doit valoir 'gaussian' ou 'student'.")
    if innovation == "student" and df <= 2.0:
        raise ValueError("Student requiert df > 2 pour que la variance existe.")

    burn = max(int(burn), 0)
    Tsim = T + burn

    # --- Localisation de la rupture (dans l'échantillon conservé puis simulé) ---
    if break_at is None:
        t_break_ret = T
    else:
        t_break_ret = int(np.clip(round(float(break_at) * T), 1, T - 1))
    t_break_sim = burn + t_break_ret
    w1 = t_break_ret / T
    w2 = 1.0 - w1

    # --- Régime 1 ---
    par1 = make_garch_params(D, persistence, alpha, uncond_vol, heterogeneous_dynamics, rng)
    R1 = make_correlation_matrix(D, corr_structure, rho, n_blocks, rho_inter, rng)

    # --- Régime 2 (identique au régime 1 en l'absence de rupture) ---
    if break_at is None:
        par2, R2 = par1, R1
        vol_mult = 1.0
    else:
        p2 = persistence if persistence_after is None else persistence_after
        a2 = alpha if alpha_after is None else alpha_after
        # Sous le modèle "common", l'amplitude est portée par h_t : on ne la
        # répercute PAS aussi sur bar_sigma, sous peine de double comptage.
        v2 = par1["uncond_vol"] if vol_model == "common" else par1["uncond_vol"] * vol_multiplier_after
        R2 = R1 if rho_after is None else make_correlation_matrix(
            D, corr_structure, rho_after, n_blocks, rho_inter, rng
        )
        par2 = make_garch_params(D, p2, a2, v2, heterogeneous_dynamics, rng)
        vol_mult = float(vol_multiplier_after)

    # --- Facteur de volatilité scalaire (modèle "common") ---
    # omega_h = (1 - p) * echelle  =>  E[h_t] = echelle, donc Sigma_true = echelle * base.
    h_scale = np.array([1.0, vol_mult**2 if vol_model == "common" else 1.0])
    p_h = np.array([persistence, persistence if persistence_after is None else persistence_after])
    a_h = np.array([alpha, alpha if alpha_after is None else alpha_after])
    b_h = p_h - a_h
    om_h = (1.0 - p_h) * h_scale

    # --- Innovations (tirées une seule fois, transformées par régime) ---
    eps = rng.standard_normal((Tsim, D))
    e_driver = rng.standard_normal(Tsim)  # driver du GARCH scalaire, INDÉPENDANT de eps
    if innovation == "student":
        g = rng.chisquare(df, size=Tsim) / df
        t_scale = np.sqrt((df - 2.0) / df)  # normalisation : Cov(z_t) = R
    L1 = np.linalg.cholesky(R1)
    L2 = np.linalg.cholesky(R2)

    # --- Boucle temporelle ---
    X = np.zeros((Tsim, D))
    cond_var = np.zeros((Tsim, D))
    s2_prev: Optional[Array] = None
    h_prev: float = 1.0

    for t in range(Tsim):
        r = 0 if t < t_break_sim else 1
        par = par1 if r == 0 else par2
        L = L1 if r == 0 else L2

        z = eps[t] @ L.T  # corrélation R du régime courant
        if innovation == "student":
            z = z * (t_scale / np.sqrt(g[t]))

        if vol_model == "ccc":
            if t == 0:
                s2 = par["uncond_var"].copy()  # initialisation stationnaire
            else:
                s2 = par["omega"] + par["alpha"] * X[t - 1] ** 2 + par["beta"] * s2_prev
            s2 = np.maximum(s2, 1e-16)  # garde-fou numérique
            X[t] = np.sqrt(s2) * z
            cond_var[t] = s2
            s2_prev = s2
        else:  # vol_model == "common"
            if t == 0:
                h = h_scale[0]
            else:
                h = om_h[r] + a_h[r] * h_prev * e_driver[t - 1] ** 2 + b_h[r] * h_prev
            h = max(float(h), 1e-16)
            X[t] = np.sqrt(h) * par["uncond_vol"] * z
            cond_var[t] = h * par["uncond_var"]
            h_prev = h

    # --- Rejet du burn-in ---
    X_out = X[burn:].copy()
    cv_out = cond_var[burn:].copy()

    # --- Vérité terrain analytique ---
    def _uncond_cov(par: Dict[str, Array], R: Array, regime: int) -> Array:
        if vol_model == "ccc":
            v = par["omega"] / (1.0 - par["alpha"] - par["beta"])  # = uncond_var
            sd = np.sqrt(v)
            S = R * np.outer(sd, sd)
        else:
            sd = par["uncond_vol"]
            S = h_scale[regime] * (R * np.outer(sd, sd))
        return 0.5 * (S + S.T)

    S1 = _uncond_cov(par1, R1, 0)
    S2 = _uncond_cov(par2, R2, 1)
    # Cible du mélange temporel (les deux régimes sont de moyenne nulle).
    Sigma_true = 0.5 * (w1 * S1 + w2 * S2 + (w1 * S1 + w2 * S2).T)

    # --- Historique des covariances conditionnelles ---
    Sigma_t: Optional[Array] = None
    if store_Sigma_t:
        Sigma_t = np.empty((T, D, D))
        for t in range(T):
            R = R1 if t < t_break_ret else R2
            sd = np.sqrt(cv_out[t])
            Sigma_t[t] = R * np.outer(sd, sd)

    d_true = np.sqrt(np.diag(Sigma_true))
    R_true = Sigma_true / np.outer(d_true, d_true)

    meta = {
        "T": T, "D": D, "vol_model": vol_model, "innovation": innovation, "df": df,
        "persistence": persistence, "alpha": alpha, "burn": burn,
        "break_at": break_at, "t_break": t_break_ret,
        "persistence_after": persistence_after, "vol_multiplier_after": vol_mult,
        "heterogeneous_dynamics": heterogeneous_dynamics,
        "corr_structure": corr_structure, "rho": rho, "rho_after": rho_after,
        "params_regime1": par1, "params_regime2": par2,
        "R_regime1": R1, "R_regime2": R2,
    }

    if output == "tuple":
        return X_out, Sigma_true

    return {
        "X": X_out,
        "Sigma_true": Sigma_true,
        "Sigma_t": Sigma_t,
        "R_true": R_true,
        "Sigma_true_regimes": (S1, S2),
        "cond_var": cv_out,
        "params": meta,
    }


# =============================================================================
# Scénarios pré-paramétrés (utilisables tels quels par les binômes 1 et 2)
# =============================================================================
SCENARIOS: Dict[str, Dict[str, Any]] = {
    # Régime modérément volatile : alpha + beta = 0.85
    "modere": dict(persistence=0.85, alpha=0.08, innovation="gaussian"),
    # Régime de crise : alpha + beta = 0.98, fort clustering de volatilité
    "crise": dict(persistence=0.98, alpha=0.12, innovation="gaussian"),
    # Cumul volatilité variable + queues lourdes (nu = 5 : moment d'ordre 4 infini)
    "crise_student": dict(persistence=0.98, alpha=0.12, innovation="student", df=5.0),
    # Choc structurel en t = T/2 : volatilité x2, persistance 0.98, contagion (rho 0.5 -> 0.75)
    "choc_structurel": dict(
        persistence=0.85, alpha=0.08, break_at=0.5,
        persistence_after=0.98, alpha_after=0.15,
        vol_multiplier_after=2.0, rho_after=0.75,
    ),
    # Régime ELLIPTIQUE exact : validation théorique de Tyler (cf. §1.3-A)
    "elliptique_commun": dict(persistence=0.98, alpha=0.12, vol_model="common"),
    # CCC à dynamiques hétérogènes : cas « difficile » pour Tyler
    "ccc_heterogene": dict(persistence=0.98, alpha=0.12, heterogeneous_dynamics=True),
}


def simulate_scenario(
    name: str,
    T: int,
    D: int,
    seed: Union[None, int, np.random.Generator] = None,
    output: str = "dict",
    **overrides: Any,
) -> Union[Dict[str, Any], Tuple[Array, Array]]:
    """
    Enveloppe de commodité : `simulate_scenario("crise", T=500, D=30, seed=0)`.
    Tout mot-clé supplémentaire écrase la valeur du préréglage.
    """
    if name not in SCENARIOS:
        raise KeyError(f"Scénario inconnu : {name!r}. Disponibles : {list(SCENARIOS)}")
    kw = dict(SCENARIOS[name])
    kw.update(overrides)
    return simulate_ccc_garch(T=T, D=D, seed=seed, output=output, **kw)