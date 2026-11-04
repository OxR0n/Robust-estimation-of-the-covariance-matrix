import numpy as np  # Bibliotheque de calcul sur les tableaux de nombres
import matplotlib

matplotlib.use("Agg")  # Enregistre les graphiques sans ouvrir de fenetre
import matplotlib.pyplot as plt  # Bibliotheque de graphiques

GRAINE = 161  # Fixe le hasard : memes nombres a chaque lancement
N = 20  # Nombre d'actifs
ALPHA = 0.15  # Reaction de la volatilite a un choc recent
BETA = 0.80  # Memoire de la volatilite passee


def creer_sigma(n, graine=GRAINE):
    """Construit la vraie matrice de covariance avec un modele a un facteur."""
    rng = np.random.default_rng(graine)  # Generateur de nombres aleatoires
    vol_marche = 0.01  # Le marche bouge d'environ 1 % par jour
    b = rng.uniform(0.5, 1.5, size=n)  # Sensibilite de chaque actif au marche
    vol_propre = rng.uniform(0.01, 0.02, size=n)  # Risque propre : 1 % a 2 %
    # Meme formule que pour Danone, Total et Air France
    return vol_marche**2 * np.outer(b, b) + np.diag(vol_propre**2)


def simuler_h(T, alpha, beta, rng, burn_in=500):
    """Simule le niveau de risque h_t du marche pendant T jours."""
    omega = 1.0 - alpha - beta  # Ce choix impose une moyenne de h egale a 1
    chocs = rng.standard_normal(T + burn_in)  # Un choc xi par jour
    h = np.empty(T + burn_in)  # Reserve la place pour toutes les valeurs
    h[0] = 1.0  # Le premier jour, on part du niveau normal

    # Chaque jour, on calcule le risque a partir du risque et du choc de la veille
    for t in range(1, T + burn_in):
        h[t] = omega + (alpha * chocs[t - 1] ** 2 + beta) * h[t - 1]

    # On jette les premiers jours pour oublier le point de depart artificiel
    return h[burn_in:]


def simuler_rendements(T, sigma, alpha, beta, rng):
    """Renvoie un tableau T x N de rendements (une ligne = un jour) et h."""
    n = sigma.shape[0]  # Nombre d'actifs
    L = np.linalg.cholesky(sigma)  # La "racine carree" de Sigma : L * L' = Sigma
    h = simuler_h(T, alpha, beta, rng)  # Le niveau de risque de chaque jour
    z = rng.standard_normal((T, n))  # Un choc independant par jour et par actif

    # z @ L.T cree les liens entre actifs (forme adaptee a un tableau en lignes)
    rendements_lies = z @ L.T

    # np.sqrt(h)[:, None] met h en colonne pour multiplier chaque ligne (chaque jour)
    return np.sqrt(h)[:, None] * rendements_lies, h


def autocorrelation(x):
    """Lien entre la valeur d'un jour et celle du lendemain."""
    x = x - x.mean()  # On retire la moyenne
    return np.sum(x[:-1] * x[1:]) / np.sum(x**2)


def kurtosis(x):
    """Vaut 3 pour une loi normale, plus de 3 si les extremes sont plus frequents."""
    x = x - x.mean()  # On retire la moyenne
    return np.mean(x**4) / np.mean(x**2) ** 2


# Ce bloc s'execute seulement quand on lance ce fichier directement
if __name__ == "__main__":
    sigma = creer_sigma(N)
    rng = np.random.default_rng(GRAINE)

    # --- Test 1 : a-t-on bien retrouve Sigma sur une tres longue simulation ? ---
    r_long, h_long = simuler_rendements(2_000_000, sigma, ALPHA, BETA, rng)
    sigma_mesuree = np.cov(r_long, rowvar=False)  # Une colonne = un actif
    erreur = np.linalg.norm(sigma_mesuree - sigma, "fro") / np.linalg.norm(sigma, "fro")
    print("Taille du tableau de rendements :", r_long.shape)
    print(f"Moyenne de h : {h_long.mean():.3f}")
    print(f"Ecart entre la covariance mesuree et Sigma : {erreur:.2%}")

    # --- Test 2 : le marche simule ressemble-t-il a un marche volatile ? ---
    actif = r_long[:, 0]  # Tous les jours du premier actif
    print(f"Autocorrelation des rendements : {autocorrelation(actif):+.4f}")
    print(f"Autocorrelation des rendements au carre : {autocorrelation(actif**2):+.4f}")
    print(f"Kurtosis du premier actif : {kurtosis(actif):.2f}")

    # --- Graphique : 1000 jours du regime ---
    r, h = simuler_rendements(1000, sigma, ALPHA, BETA, rng)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(r[:, 0] * 100, lw=0.7, color="#1f4e79")
    axes[0].set_ylabel("Rendement de l'actif 1 (%)")
    axes[0].set_title("Marche tres volatile : les fortes variations arrivent en grappes")
    axes[1].plot(np.sqrt(h), lw=1, color="#c0392b")
    axes[1].axhline(1, color="grey", ls="--", lw=0.8)  # Le niveau normal
    axes[1].set_ylabel("Multiplicateur de volatilite")
    axes[1].set_xlabel("Jour")
    fig.tight_layout()
    fig.savefig("garch_tyler/regime_garch.png", dpi=150)
    print("Graphique enregistre : garch_tyler/regime_garch.png")