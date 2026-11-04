import numpy as np 


# Fixe le hasard : memes nombres a chaque lancement
GRAINE = 161  

# Nombre d'actifs
N = 20  


def creer_sigma(n, graine=GRAINE):
    """Construit la vraie matrice de covariance avec un modele a un facteur."""
    rng = np.random.default_rng(graine)
    vol_marche = 0.01
    b = rng.uniform(0.5, 1.5, size=n)
    vol_propre = rng.uniform(0.01, 0.02, size=n)

    return vol_marche**2 * np.outer(b, b) + np.diag(vol_propre**2)


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