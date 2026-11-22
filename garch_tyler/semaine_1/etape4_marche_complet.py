import numpy as np

GRAINE = 161
N = 20
VOL_MARCHE = 0.01



ALPHA = 0.15
BETA = 0.80
OMEGA = 1 - ALPHA - BETA


def creer_actions(n, graine=GRAINE):
    """Invente n actions : leur sensibilite au marche et leur risque propre."""
    rng = np.random.default_rng(graine)
    b = rng.uniform(0.5, 1.5, size=n)
    propre = rng.uniform(0.01, 0.02, size=n)
    return b, propre


def vraie_sigma(b, propre):
    """Le corrige : le tableau 20 x 20 des variances et covariances (etape 2)."""
    return VOL_MARCHE**2 * np.outer(b, b) + np.diag(propre**2)


def simuler_marche(nb_jours, b, propre, rng, jours_jetes=500):
    """Fabrique nb_jours de rendements pour toutes les actions."""
    total = nb_jours + jours_jetes
    choc_marche = rng.standard_normal(total)
    chocs_propres = rng.standard_normal((total, len(b)))

    # ici on calcule la nervosite h du marche, qui va influencer tous les rendements
    h = np.empty(total)
    h[0] = 1.0
    for t in range(1, total):
        h[t] = OMEGA + ALPHA * choc_marche[t - 1] ** 2 * h[t - 1] + BETA * h[t - 1]

    # Rendement sans nervosite = part du marche + part propre (etape 2)
    # [:, None] transforme la liste des jours en colonne, pour la combiner avec chaque action
    part_marche = VOL_MARCHE * choc_marche[:, None] * b
    part_propre = chocs_propres * propre

    # La nervosite touche tout
    # h agit sur la variance. Les rendements, eux, sont à l’échelle de l’écart-type, qui est la racine de la variance.
    rendements = np.sqrt(h)[:, None] * (part_marche + part_propre)

    # On jette les premiers jours, ou la nervosite partait d'une valeur arbitraire
    return rendements[jours_jetes:], h[jours_jetes:]


if __name__ == "__main__":
    b, propre = creer_actions(N)
    sigma = vraie_sigma(b, propre)
    vols = np.sqrt(np.diag(sigma))
    correlations = sigma / np.outer(vols, vols)
    paires = correlations[np.triu_indices(N, k=1)]

    print("--- Le corrige Sigma ---")
    print("Taille :", sigma.shape)
    print(f"Volatilites : {vols.min():.2%} a {vols.max():.2%}")
    print(f"Correlations : {paires.min():.2f} a {paires.max():.2f}")

    rng = np.random.default_rng(GRAINE)
    rendements, h = simuler_marche(1000, b, propre, rng)
    print("\n--- Les donnees simulees ---")
    print("Taille du tableau de rendements :", rendements.shape)

    print("\n10 premiers jours, 4 premieres actions :")
    print("  Jour  Nervosite   Action 1  Action 2  Action 3  Action 4")
    for t in range(100):
        valeurs = "  ".join(f"{r:+7.2%}" for r in rendements[t, :4])
        print(f"  {t + 1:4d}    {h[t]:.2f}      {valeurs}")

    # Comparer les jours nerveux et les jours calmes

    # Les seuils ne sont pas choisis a la main : on les calcule a partir des donnees
    # np.quantile(h, 0.8) = la nervosite que seuls 20 % des jours depassent
    # np.quantile(h, 0.2) = la nervosite sous laquelle se trouvent 20 % des jours
    seuil_nerveux = np.quantile(h, 0.8)
    seuil_calme = np.quantile(h, 0.2)

    # Jours ou la nervosite depasse le seuil : les 20 % de jours les plus nerveux
    nerveux = h > seuil_nerveux

    # Jours ou la nervosite est sous le seuil : les 20 % de jours les plus calmes
    calmes = h < seuil_calme

    print(f"\nJours nerveux (20 % les plus nerveux, h > {seuil_nerveux:.2f}) : {nerveux.sum()} jours")
    print(f"  Mouvements de plus de 3 % : {np.mean(np.abs(rendements[nerveux]) > 0.03):.0%}")
    print(f"Jours calmes (20 % les plus calmes, h < {seuil_calme:.2f}) : {calmes.sum()} jours")
    print(f"  Mouvements de plus de 3 % : {np.mean(np.abs(rendements[calmes]) > 0.03):.0%}")