"""
Estimateur de covariance de type Catoni.

Idee : Sigma_ij = E[(x_i - mu_i)(x_j - mu_j)] est une MOYENNE. On l'estime avec
l'estimateur de Catoni de la moyenne, applique au produit centre de chaque paire.
Le centrage (mu_i) utilise lui aussi l'estimateur de Catoni.
La matrice obtenue n'est pas forcement semi-definie positive : on la projette
sur le cone des matrices SDP (ecretage des valeurs propres).
"""
import numpy as np

try:
    import pandas as pd
except ImportError:      # pandas facultatif
    pd = None


# ------------------------------------------------------------ fonction d'influence
def _psi(x):
    """psi(x) = log(1 + x + x^2/2) si x >= 0, -log(1 - x + x^2/2) si x < 0.
    Croissante, impaire, croissance seulement logarithmique."""
    return np.sign(x) * np.log1p(np.abs(x) + 0.5 * x**2)


# ------------------------------------------------------- moyenne de Catoni (vectorisee)
def _catoni_mean_block(Y, delta=0.05, n_iter=100):
    """Estimateur de Catoni de la moyenne, colonne par colonne.

    Y : (n, k). Renvoie theta de taille k, solution de
        sum_t psi( alpha * (Y_t - theta) ) = 0.
    alpha = sqrt( 2 log(1/delta) / (n * v) ), v = variance empirique de la colonne.
    La fonction theta -> sum psi(.) est strictement decroissante : dichotomie.
    """
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0]
    v = Y.var(axis=0, ddof=1)
    v = np.where(v > 0, v, 1.0)                       # colonne constante : evite 1/0
    alpha = np.sqrt(2.0 * np.log(1.0 / delta) / (n * v))

    lo, hi = Y.min(axis=0), Y.max(axis=0)             # theta est entre min et max
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f = _psi(alpha * (Y - mid)).sum(axis=0)
        lo = np.where(f > 0, mid, lo)                 # f > 0 : theta est plus grand
        hi = np.where(f > 0, hi, mid)
    return 0.5 * (lo + hi)


def catoni_mean(Y, delta=0.05, n_iter=100, max_elems=3_000_000):
    """Idem, mais traite les colonnes par blocs pour limiter la memoire
    (utile quand n est tres grand, par ex. 100 000 dates)."""
    Y = np.asarray(Y, dtype=float)
    step = max(1, max_elems // Y.shape[0])
    return np.concatenate([_catoni_mean_block(Y[:, k:k + step], delta, n_iter)
                           for k in range(0, Y.shape[1], step)])


# --------------------------------------------------------------- projection SDP
def nearest_psd(S, eps=1e-10):
    """Projection de Frobenius sur le cone SDP : on met a eps * max les valeurs
    propres trop petites ou negatives."""
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, eps * w.max())
    return (V * w) @ V.T


# ------------------------------------------------------ covariance de Catoni
def catoni_covariance(X, delta=0.05, psd=True, return_info=False):
    """Matrice de covariance robuste par estimateur de Catoni.

    X : (n, p) tableau numpy ou DataFrame pandas (une colonne par actif).
    delta : niveau de confiance de l'estimateur, dans ]0, 1[ (alpha = sqrt(2 log(1/delta)/(n v))).
            delta PETIT -> alpha GRAND -> psi comprime beaucoup -> estimateur
            tres robuste (proche de la mediane) ;
            delta GRAND (vers 1) -> alpha petit -> psi quasi lineaire -> on se
            rapproche de la moyenne empirique.
    psd   : si True, projette le resultat sur les matrices semi-definies positives.
    Renvoie Sigma_hat (meme type que X : DataFrame si X en est un).
    """
    is_df = pd is not None and isinstance(X, pd.DataFrame)
    cols = X.columns if is_df else None
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n < 3:
        raise ValueError("Il faut au moins 3 observations.")

    # 1) centrage robuste
    mu = catoni_mean(X, delta)
    Xc = X - mu

    # 2) produits centres de toutes les paires i <= j : (n, p(p+1)/2)
    iu, ju = np.triu_indices(p)
    Y = Xc[:, iu] * Xc[:, ju]

    # 3) Catoni sur chaque paire
    theta = catoni_mean(Y, delta)
    S = np.zeros((p, p))
    S[iu, ju] = theta
    S[ju, iu] = theta                                 # symetrie

    # 4) correction SDP
    n_neg = int((np.linalg.eigvalsh(S) < 0).sum())
    if psd:
        S = nearest_psd(S)

    if is_df:
        S = pd.DataFrame(S, index=cols, columns=cols)
    return (S, {"mu": mu, "n_negative_eigenvalues": n_neg}) if return_info else S
