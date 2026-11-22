import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Permet d'importer les fichiers des dossiers voisins semaine_1 et semaine_2
sys.path.append(str(Path(__file__).resolve().parent.parent / "semaine_1"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "semaine_2"))
import etape4_marche_complet as marche
from etape4_marche_complet import GRAINE, N, creer_actions, simuler_marche, vraie_sigma
from etape2_normes_erreur import erreurs_relatives
from etape1_tyler import tyler_forme, forme_de

# Nombres de jours observes testes (Tyler a besoin de plus de jours que d'actions)
TAILLES_T = [50, 100, 250, 500, 1000]

# Nombre de copies notees pour chaque T
NB_REPETITIONS = 500

# Les deux marches compares : (alpha, beta, omega)
REGIMES = {
    # Nervosite fixe a 1
    "Gaussien": (0.0, 0.0, 1.0),
    # Notre marche tres volatile
    "GARCH": (0.15, 0.80, 0.05),
}

if __name__ == "__main__":
    b, propre = creer_actions(N)
    sigma = vraie_sigma(b, propre)

    # On compare des FORMES : c'est ce qui decide des poids du portefeuille de variance minimale
    forme_vraie = forme_de(sigma)
    rng = np.random.default_rng(GRAINE)

    # resultats[regime][estimateur] = liste de (Frobenius moyen, spectrale moyenne), une par T
    resultats = {}

    for nom, (alpha, beta, omega) in REGIMES.items():
        # On regle le marche
        marche.ALPHA, marche.BETA, marche.OMEGA = alpha, beta, omega
        resultats[nom] = {"Empirique": [], "Tyler": []}

        for T in TAILLES_T:
            notes_emp = []
            notes_tyl = []
            for _ in range(NB_REPETITIONS):
                # Un nouvel echantillon : les deux estimateurs recoivent les MEMES donnees
                rendements, _ = simuler_marche(T, b, propre, rng)

                # Copie de l'empirique, ramenee a sa forme
                forme_emp = forme_de(np.cov(rendements, rowvar=False))
                notes_emp.append(erreurs_relatives(forme_emp, forme_vraie))

                # Copie de Tyler (deja une forme)
                forme_tyl, _ = tyler_forme(rendements)
                notes_tyl.append(erreurs_relatives(forme_tyl, forme_vraie))

            # Moyenne des 500 notes : (Frobenius, spectrale)
            resultats[nom]["Empirique"].append(np.mean(notes_emp, axis=0))
            resultats[nom]["Tyler"].append(np.mean(notes_tyl, axis=0))

    # Le tableau : erreur sur la forme, et gain de Tyler
    print(f"Erreur moyenne sur la forme de Sigma ({NB_REPETITIONS} copies par case)")
    print("Gain = de combien Tyler reduit l'erreur de l'empirique (negatif = Tyler pire)\n")
    for nom in REGIMES:
        print(f"--- Marche {nom} ---")
        print("              Frobenius                      Spectrale")
        print("    T   Empirique   Tyler    Gain     Empirique   Tyler    Gain")
        for k, T in enumerate(TAILLES_T):
            fe, se = resultats[nom]["Empirique"][k]
            ft, st = resultats[nom]["Tyler"][k]
            print(f"{T:5d}     {fe:6.1%}   {ft:6.1%}   {1 - ft / fe:+5.0%}"
                  f"       {se:6.1%}   {st:6.1%}   {1 - st / se:+5.0%}")
        print()

    # Graphique : l'erreur de Frobenius sur la forme selon T
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = [
        ("Gaussien", "Empirique", "#1f4e79", "o", "-"),
        ("Gaussien", "Tyler", "#1f4e79", "o", "--"),
        ("GARCH", "Empirique", "#c0392b", "s", "-"),
        ("GARCH", "Tyler", "#c0392b", "s", "--"),
    ]
    for regime, estimateur, couleur, marque, trait in styles:
        erreurs = [r[0] for r in resultats[regime][estimateur]]
        ax.loglog(TAILLES_T, erreurs, marque + trait, color=couleur, label=f"{estimateur} - {regime}")
    ax.set_xlabel("Nombre de jours observes T")
    ax.set_ylabel("Erreur relative de Frobenius sur la forme")
    ax.set_title(f"Tyler contre empirique, {N} actions")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("tyler_vs_empirique.png", dpi=150)
    print("Graphique enregistre : tyler_vs_empirique.png")