import sys
from pathlib import Path

import numpy as np  
import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt  

# Permet d'importer les fichiers du dossier voisin semaine 1
sys.path.append(str(Path(__file__).resolve().parent.parent / "semaine_1"))
import etape4_marche_complet as marche
from etape4_marche_complet import GRAINE, N, creer_actions, simuler_marche, vraie_sigma
from etape2_normes_erreur import erreurs_relatives

# Nombres de jours observes testes
TAILLES_T = [50, 100, 250, 500, 1000]  

# Nombre de copies notees pour chaque T
NB_REPETITIONS = 500  

# Les deux marches compares : (alpha, beta, omega)
REGIMES = {
    # Nervosite fixe a 1
    "Gaussien": (0.0, 0.0, 1.0),  
    # Ton marche tres volatile
    "GARCH": (0.15, 0.80, 0.05),  
}

if __name__ == "__main__":
    b, propre = creer_actions(N)
    sigma = vraie_sigma(b, propre)
    rng = np.random.default_rng(GRAINE)
    # resultats[regime] = liste de (Frobenius moyen, spectrale moyenne)
    resultats = {}  

    for nom, (alpha, beta, omega) in REGIMES.items():
        # On regle le marche
        marche.ALPHA, marche.BETA, marche.OMEGA = alpha, beta, omega  
        resultats[nom] = []
        for T in TAILLES_T:
            notes = [erreurs_relatives(np.cov(simuler_marche(T, b, propre, rng)[0], rowvar=False), sigma)
                     for _ in range(NB_REPETITIONS)]
            # Moyenne des 500 notes
            resultats[nom].append(np.mean(notes, axis=0))  

    print(f"Erreur moyenne de la covariance empirique ({NB_REPETITIONS} copies par case)\n")
    print("            Frobenius               Spectrale")
    print("    T   Gaussien   GARCH   Ratio   Gaussien   GARCH   Ratio")
    for k, T in enumerate(TAILLES_T):
        fg, sg = resultats["Gaussien"][k]
        fv, sv = resultats["GARCH"][k]
        print(f"{T:5d}    {fg:6.1%}  {fv:6.1%}   x{fv / fg:.2f}    {sg:6.1%}  {sv:6.1%}   x{sv / sg:.2f}")

    # Combien de jours gaussiens faudrait-il pour la meme erreur que 1000 jours GARCH ?
    fg, _ = resultats["Gaussien"][-1]
    fv, _ = resultats["GARCH"][-1]
    # L'erreur suit 1 / racine(T)
    equivalent = 1000 * (fg / fv) ** 2  
    print(f"\n1000 jours de marche volatile valent environ {equivalent:.0f} jours de marche gaussien")

    # Graphique : l'erreur de Frobenius selon T
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for nom, couleur, marque in [("Gaussien", "#1f4e79", "o"), ("GARCH", "#c0392b", "s")]:
        ax.loglog(TAILLES_T, [r[0] for r in resultats[nom]], marque + "-", color=couleur, label=nom)
    ax.set_xlabel("Nombre de jours observes T")
    ax.set_ylabel("Erreur relative de Frobenius")
    ax.set_title(f"Covariance empirique, {N} actions")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("erreur_selon_T.png", dpi=150)
    print("Graphique enregistre : erreur_selon_T.png")