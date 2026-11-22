import sys
from pathlib import Path

import numpy as np  

# Permet d'importer les fichiers du dossier voisin semaine_1
sys.path.append(str(Path(__file__).resolve().parent.parent / "semaine_1"))
import etape4_marche_complet as marche
from etape4_marche_complet import GRAINE, N, creer_actions, simuler_marche, vraie_sigma


def erreurs_relatives(sigma_chapeau, sigma):
    """Note la copie : erreur relative de Frobenius et erreur relative spectrale."""
    # Le tableau des erreurs, case par case
    difference = sigma_chapeau - sigma  
    frobenius = np.linalg.norm(difference, "fro") / np.linalg.norm(sigma, "fro")
    spectrale = np.linalg.norm(difference, 2) / np.linalg.norm(sigma, 2)
    return frobenius, spectrale


if __name__ == "__main__":
    # Partie 1 : le petit exemple calcule a la main
    # Le corrige
    sigma_exemple = np.array([[3.0, 0.0], [0.0, 2.0]])  
    # La copie
    chapeau_exemple = np.array([[4.0, 1.0], [1.0, 1.0]])  
    difference = chapeau_exemple - sigma_exemple

    print("=== Petit exemple ===")
    print("Tableau des erreurs :")
    print(difference)

    # Frobenius : racine de la somme des carres de toutes les cases
    print(f"Frobenius a la main  : {np.sqrt(np.sum(difference**2)):.3f}")
    print(f"Frobenius par numpy  : {np.linalg.norm(difference, 'fro'):.3f}")

    # Spectrale : la plus grande valeur propre, en valeur absolue
    print(f"Valeurs propres de l'erreur : {np.linalg.eigvalsh(difference)}")
    print(f"Spectrale a la main  : {np.max(np.abs(np.linalg.eigvalsh(difference))):.3f}")
    print(f"Spectrale par numpy  : {np.linalg.norm(difference, 2):.3f}")

    f, s = erreurs_relatives(chapeau_exemple, sigma_exemple)
    print(f"Erreurs relatives : Frobenius {f:.0%}, spectrale {s:.0%}")

    # Partie 2 : les 3 echantillons de l'etape 1 (marche gaussien)
    # Nervosite fixe a 1
    marche.ALPHA, marche.BETA, marche.OMEGA = 0.0, 0.0, 1.0  
    b, propre = creer_actions(N)
    sigma = vraie_sigma(b, propre)
    # Meme graine : memes echantillons qu'a l'etape 1
    rng = np.random.default_rng(GRAINE)  
    T = 250

    print(f"\n=== 20 actions, {T} jours, marche gaussien ===")
    for essai in range(1, 4):
        rendements, _ = simuler_marche(T, b, propre, rng)
        f, s = erreurs_relatives(np.cov(rendements, rowvar=False), sigma)
        print(f"Echantillon {essai} : Frobenius {f:.1%}, spectrale {s:.1%}")