import numpy as np  # Bibliotheque de calcul sur les tableaux de nombres

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


# Ce bloc s'execute seulement quand on lance ce fichier directement
if __name__ == "__main__":
    sigma = creer_sigma(N)

    # La volatilite est la racine de la variance, qui est sur la diagonale
    vols = np.sqrt(np.diag(sigma))

    # Correlation = covariance / (volatilite i * volatilite j)
    correlations = sigma / np.outer(vols, vols)

    # On garde les cases au-dessus de la diagonale : chaque paire une seule fois
    hors_diagonale = correlations[np.triu_indices(N, k=1)]

    # eigvalsh calcule les valeurs propres d'une matrice symetrique
    valeurs_propres = np.linalg.eigvalsh(sigma)

    print("Taille de Sigma :", sigma.shape)
    print("Symetrique :", np.allclose(sigma, sigma.T))
    print(f"Plus petite valeur propre : {valeurs_propres.min():.3e}")
    print(f"Plus grande valeur propre : {valeurs_propres.max():.3e}")
    print(f"Volatilites : {vols.min():.2%} a {vols.max():.2%}")
    print(f"Correlations : {hors_diagonale.min():.2f} a {hors_diagonale.max():.2f}")

    # --- Controle du GARCH ---
    rng = np.random.default_rng(GRAINE)
    h = simuler_h(1000, ALPHA, BETA, rng)  # Environ 4 ans de Bourse
    multiplicateur = np.sqrt(h)  # Ce qui multiplie la volatilite chaque jour

    print("\n--- Variance GARCH sur 1000 jours ---")
    print("10 premieres valeurs de h :", np.round(h[:10], 2))
    print(f"Moyenne de h : {h.mean():.3f}")
    print(f"Multiplicateur de volatilite : {multiplicateur.min():.2f} a {multiplicateur.max():.2f}")
    print(f"Part des jours avec volatilite multipliee par plus de 1,5 : {np.mean(multiplicateur > 1.5):.1%}")

    # Sur une tres longue periode, la moyenne doit se rapprocher de 1
    h_long = simuler_h(1_000_000, ALPHA, BETA, rng)
    print(f"Moyenne de h sur 1 000 000 de jours : {h_long.mean():.3f}")