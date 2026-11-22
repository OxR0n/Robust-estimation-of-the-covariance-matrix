import sys
from pathlib import Path

import numpy as np

# Permet d'importer les fichiers des dossiers voisins semaine_1 et semaine_2
sys.path.append(str(Path(__file__).resolve().parent.parent / "semaine_1"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "semaine_2"))
import etape4_marche_complet as marche
from etape4_marche_complet import GRAINE, N, creer_actions, simuler_marche, vraie_sigma
from etape2_normes_erreur import erreurs_relatives


def tyler_forme(rendements, tolerance=1e-6, max_iterations=200):
    """Estimateur de Tyler : retrouve la FORME de Sigma (pas sa taille).

    Idee : chaque jour est ramene a la meme taille avant d'etre compte,
    pour qu'un jour de crise ne pese pas plus qu'un jour calme.
    Renvoie la forme (trace = nombre d'actions) et le nombre d'iterations."""

    # On centre les rendements (comme np.cov)
    X = rendements - rendements.mean(axis=0)
    T, n = X.shape

    # Point de depart : toutes les actions identiques et independantes
    forme = np.eye(n)

    for iteration in range(1, max_iterations + 1):
        # Taille de chaque jour, mesuree avec la forme actuelle
        # (distance de Mahalanobis au carre : un jour de crise donne un grand nombre)
        inverse = np.linalg.inv(forme)
        taille_jour = np.sum((X @ inverse) * X, axis=1)

        # Chaque jour est divise par sa taille : tous les jours pesent pareil
        X_ramene = X / taille_jour[:, None]

        # Nouvelle forme = moyenne des jours ramenes (meme calcul que np.cov)
        nouvelle = (n / T) * (X_ramene.T @ X)

        # Tyler ne connait que la forme : on fixe sa taille (trace = n)
        nouvelle = n * nouvelle / np.trace(nouvelle)

        # On s'arrete quand la forme ne bouge presque plus
        changement = np.linalg.norm(nouvelle - forme, "fro") / np.linalg.norm(forme, "fro")
        forme = nouvelle
        if changement < tolerance:
            break

    return forme, iteration


def tyler(rendements):
    """Tyler complet : la forme de Tyler, remise a la bonne taille.
    La taille (trace) est prise sur la covariance empirique."""
    forme, _ = tyler_forme(rendements)
    taille = np.trace(np.cov(rendements, rowvar=False))
    return forme * taille / forme.shape[0]


def forme_de(sigma):
    """La forme d'une matrice : on la remet a la taille trace = nombre d'actions."""
    return sigma.shape[0] * sigma / np.trace(sigma)


if __name__ == "__main__":
    # Marche GARCH (les vrais reglages)
    marche.ALPHA, marche.BETA, marche.OMEGA = 0.15, 0.80, 0.05
    b, propre = creer_actions(N)
    sigma = vraie_sigma(b, propre)
    rng = np.random.default_rng(GRAINE)

    # --- Verification 1 : l'algorithme converge et donne une vraie forme ---
    rendements, _ = simuler_marche(250, b, propre, rng)
    forme, nb_iterations = tyler_forme(rendements)
    print("=== Verification 1 : l'algorithme (250 jours GARCH) ===")
    print(f"Converge en {nb_iterations} iterations")
    print("Symetrique :", np.allclose(forme, forme.T))
    print(f"Trace : {np.trace(forme):.2f} (doit valoir {N})")
    print(f"Plus petite valeur propre : {np.linalg.eigvalsh(forme).min():.3f} (doit etre > 0)")

    # --- Verification 2 : avec beaucoup de jours, Tyler retrouve la forme de Sigma ---
    rendements, _ = simuler_marche(100_000, b, propre, rng)
    f, s = erreurs_relatives(tyler_forme(rendements)[0], forme_de(sigma))
    print("\n=== Verification 2 : 100 000 jours GARCH ===")
    print(f"Erreur sur la forme : Frobenius {f:.2%}, spectrale {s:.2%} (doit etre petit)")

    # --- Apercu : Tyler contre l'empirique, sur un an de marche GARCH ---
    print("\n=== Apercu : 250 jours GARCH, 3 echantillons ===")
    print("                Erreur sur la forme           Erreur complete (Frobenius)")
    print("Echantillon   Empirique   Tyler              Empirique   Tyler")
    for essai in range(1, 4):
        rendements, _ = simuler_marche(250, b, propre, rng)
        empirique = np.cov(rendements, rowvar=False)
        forme_emp, _ = erreurs_relatives(forme_de(empirique), forme_de(sigma))
        forme_tyl, _ = erreurs_relatives(tyler_forme(rendements)[0], forme_de(sigma))
        complet_emp, _ = erreurs_relatives(empirique, sigma)
        complet_tyl, _ = erreurs_relatives(tyler(rendements), sigma)
        print(f"     {essai}         {forme_emp:6.1%}    {forme_tyl:6.1%}"
              f"             {complet_emp:6.1%}    {complet_tyl:6.1%}")