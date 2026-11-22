import sys
from pathlib import Path

import numpy as np  

# Permet d'importer les fichiers du dossier voisin semaine_1
sys.path.append(str(Path(__file__).resolve().parent.parent / "semaine_1"))
import etape4_marche_complet as marche
from etape4_marche_complet import GRAINE, N, creer_actions, simuler_marche, vraie_sigma


# Partie 1 : le petit exemple calcule a la main ---
exemple = np.array([[2.0, 1.0],   # Jour 1 : action A, action B (en %)
                    [-1.0, 0.0],  # Jour 2
                    [0.0, -1.0],  # Jour 3
                    [3.0, 4.0]])  # Jour 4
print("=== Petit exemple : 2 actions, 4 jours ===")
print(np.cov(exemple, rowvar=False))  

# Partie 2 : le marche de la semaine 1, en version calme (sans nervosite) ---
# Nervosite fixe a 1 : marche gaussien
marche.ALPHA, marche.BETA, marche.OMEGA = 0.0, 0.0, 1.0  

b, propre = creer_actions(N)
sigma = vraie_sigma(b, propre)
rng = np.random.default_rng(GRAINE)

# Nombre de jours observes : environ une annee de Bourse
T = 250  

print(f"\n=== 20 actions, {T} jours, 3 echantillons differents ===")
print("Case du tableau (x 10 000)        Corrige   Ech. 1   Ech. 2   Ech. 3")
estimations = []
for essai in range(3):
    # Un nouvel echantillon de T jours
    rendements, _ = simuler_marche(T, b, propre, rng) 
    # La copie : Sigma chapeau
    estimations.append(np.cov(rendements, rowvar=False))  

for nom, i, j in [("Variance action 1", 0, 0), ("Variance action 2", 1, 1), ("Covariance actions 1 et 2", 0, 1)]:
    valeurs = "   ".join(f"{e[i, j] * 1e4:5.2f}" for e in estimations)
    print(f"{nom:32s}  {sigma[i, j] * 1e4:5.2f}    {valeurs}")