import sys
from pathlib import Path

import numpy as np

# Permet d'importer les fichiers du dossier voisin semaine 1
sys.path.append(str(Path(__file__).resolve().parent.parent / "semaine_1"))
import etape4_marche_complet as marche
from etape4_marche_complet import GRAINE, N, creer_actions, simuler_marche, vraie_sigma

# Nombres de jours observes testes
TAILLES_T = [50, 100, 250, 500, 1000]

# Nombre de copies notees pour chaque T
NB_REPETITIONS = 500  


def erreur_theorique(sigma, T):
    """Erreur de Frobenius moyenne predite par la theorie pour un marche gaussien.
    Formule exacte : moyenne de l'erreur au carre = (trace^2 + Frobenius^2) / (T - 1)."""

    # Somme des variances (la diagonale)
    trace = np.trace(sigma)  

    # Somme des carres de toutes les cases
    frobenius2 = np.sum(sigma**2)  
    return np.sqrt((trace**2 + frobenius2) / (T - 1))


if __name__ == "__main__":

    # Marche gaussien : nervosite fixe a 1
    marche.ALPHA, marche.BETA, marche.OMEGA = 0.0, 0.0, 1.0  
    b, propre = creer_actions(N)
    sigma = vraie_sigma(b, propre)

    # Pour passer en erreur relative
    taille_sigma = np.linalg.norm(sigma, "fro")  
    rng = np.random.default_rng(GRAINE)

    print(f"Marche gaussien, {N} actions, {NB_REPETITIONS} copies pour chaque T\n")
    print("    T    Frobenius simule   Frobenius theorie   Ecart   Spectrale simulee")
    for T in TAILLES_T:
        erreurs_fro = []
        erreurs_spec = []
        for _ in range(NB_REPETITIONS):

            # Un nouvel echantillon
            rendements, _ = simuler_marche(T, b, propre, rng)

            # Copie - corrige  
            difference = np.cov(rendements, rowvar=False) - sigma  
            erreurs_fro.append(np.linalg.norm(difference, "fro"))
            erreurs_spec.append(np.linalg.norm(difference, 2) / np.linalg.norm(sigma, 2))

        # Moyenne quadratique : racine de la moyenne des carres, comme dans la theorie
        simule = np.sqrt(np.mean(np.square(erreurs_fro)))
        theorie = erreur_theorique(sigma, T)
        print(f"{T:5d}        {simule / taille_sigma:6.1%}             {theorie / taille_sigma:6.1%}"
              f"        {simule / theorie - 1:+5.1%}       {np.mean(erreurs_spec):6.1%}")