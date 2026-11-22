import numpy as np  
import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# On reutilise le marche construit a l'etape 4
import etape4_marche_complet as marche
from etape4_marche_complet import GRAINE, N, creer_actions, simuler_marche, vraie_sigma


def autocorrelation(x):
    """Lien entre la valeur d'un jour et celle du lendemain (de -1 a 1)."""
    x = x - x.mean()  
    return np.sum(x[:-1] * x[1:]) / np.sum(x**2)


def kurtosis(x):
    """Vaut 3 pour une loi normale, plus de 3 si les jours extremes sont plus frequents."""
    x = x - x.mean()  
    return np.mean(x**4) / np.mean(x**2) ** 2


b, propre = creer_actions(N)
sigma = vraie_sigma(b, propre)
rng = np.random.default_rng(GRAINE)

# Verification 1 : le corrige est une vraie matrice de covariance
print("=== Verification 1 : Sigma ===")
print("Symetrique :", np.allclose(sigma, sigma.T))

# Aucun portefeuille ne peut avoir un risque nul ou negatif, sinon pourquoi venir a l'école.
print(f"Plus petite valeur propre : {np.linalg.eigvalsh(sigma).min():.3e} (doit etre > 0)")

# Verification 2 : sur tres longtemps, les donnees retrouvent le corrige 
# Environ 2000 ans de Bourse
NB_JOURS = 500_000  
r_volatil, h = simuler_marche(NB_JOURS, b, propre, rng)

# Covariance calculee sur les donnees
sigma_mesuree = np.cov(r_volatil, rowvar=False)  
ecart = np.linalg.norm(sigma_mesuree - sigma) / np.linalg.norm(sigma)
print("\n=== Verification 2 : les donnees visent le bon corrige ===")
print(f"Nervosite moyenne : {h.mean():.3f} (doit etre proche de 1)")
print(f"Ecart entre la covariance mesuree et Sigma : {ecart:.2%} (doit etre petit)")


# Verification 3 : le marche se comporte comme un marche tres volatile
# Marche de comparaison : memes actions, mais sans nervosite (alpha = beta = 0)
marche.ALPHA, marche.BETA, marche.OMEGA = 0.0, 0.0, 1.0
r_calme, _ = simuler_marche(NB_JOURS, b, propre, rng)

# On remet les vrais reglages
marche.ALPHA, marche.BETA, marche.OMEGA = 0.15, 0.80, 0.05  

print("\n=== Verification 3 : comportement de l'action 1 ===")
print("                                  Marche calme   Marche volatile")
print(f"Autocorrelation des rendements   : {autocorrelation(r_calme[:, 0]):+.3f}         {autocorrelation(r_volatil[:, 0]):+.3f}")
print(f"Autocorrelation des rendements^2 : {autocorrelation(r_calme[:, 0] ** 2):+.3f}         {autocorrelation(r_volatil[:, 0] ** 2):+.3f}")
print(f"Kurtosis                         : {kurtosis(r_calme[:, 0]):.2f}           {kurtosis(r_volatil[:, 0]):.2f}")

# Verification 4 : le graphique, sur 1000 jours
r, h_vue = simuler_marche(1000, b, propre, rng)
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
axes[0].plot(r[:, 0] * 100, lw=0.7, color="#1f4e79")
axes[0].set_ylabel("Rendement de l'action 1 (%)")
axes[0].set_title("Marche tres volatile : les fortes variations arrivent en grappes")
axes[1].plot(np.sqrt(h_vue), lw=1, color="#c0392b")

# Le niveau normal
axes[1].axhline(1, color="grey", ls="--", lw=0.8)  
axes[1].set_ylabel("Multiplicateur de volatilite")
axes[1].set_xlabel("Jour")
fig.tight_layout()
fig.savefig("regime_garch.png", dpi=150)
print("\nGraphique enregistre : regime_garch.png")