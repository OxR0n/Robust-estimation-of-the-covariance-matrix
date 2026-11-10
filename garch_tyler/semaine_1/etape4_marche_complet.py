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

    h = np.empty(total)
    h[0] = 1.0
    for t in range(1, total):
        h[t] = OMEGA + ALPHA * choc_marche[t - 1] ** 2 * h[t - 1] + BETA * h[t - 1]

    # Rendement sans nervosite = part du marche + part propre (etape 2)
    # [:, None] transforme la liste des jours en colonne, pour la combiner avec chaque action
    part_marche = VOL_MARCHE * choc_marche[:, None] * b
    part_propre = chocs_propres * propre

    # La nervosite touche tout
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

    print("\n5 premiers jours, 4 premieres actions :")
    print("  Jour  Nervosite   Action 1  Action 2  Action 3  Action 4")
    for t in range(5):
        valeurs = "  ".join(f"{r:+7.2%}" for r in rendements[t, :4])
        print(f"  {t + 1:4d}    {h[t]:.2f}      {valeurs}")

    # Comparer les jours nerveux et les jours calmes

    # Jours ou la nervosite depasse 1,5
    nerveux = h > 1.5 

    # Jours ou la nervosite est sous 0,7 
    calmes = h < 0.7  

    print(f"\nJours nerveux (h > 1,5) : {nerveux.sum()} jours")
    print(f"  Mouvements de plus de 3 % : {np.mean(np.abs(rendements[nerveux]) > 0.03):.0%}")
    print(f"Jours calmes (h < 0,7) : {calmes.sum()} jours")
    print(f"  Mouvements de plus de 3 % : {np.mean(np.abs(rendements[calmes]) > 0.03):.0%}")