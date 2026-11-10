import numpy as np  

rng = np.random.default_rng(161)

VOL_NORMALE = 0.02

# Reaction : a quel point un gros choc hier rend le marche nerveux aujourd'hui
ALPHA = 0.15  

# Memoire : a quel point la nervosite d'hier reste aujourd'hui
BETA = 0.80  

# Rappel vers la normale (0,05) : la nervosite vaut 1 en moyenne
OMEGA = 1 - ALPHA - BETA  

NB_JOURS = 1000
chocs = rng.standard_normal(NB_JOURS)  

# --- Marche 1 : risque constant ---
calme = VOL_NORMALE * chocs

# --- Marche 2 : risque qui change, avec un thermostat de nervosite h ---
h = np.empty(NB_JOURS)  
h[0] = 1.0  
for t in range(1, NB_JOURS):

    # Nervosite d'aujourd'hui = rappel + reaction au choc d'hier + memoire d'hier
    h[t] = OMEGA + ALPHA * chocs[t - 1] ** 2 * h[t - 1] + BETA * h[t - 1]

# Memes nouvelles, sensibilite variable
volatile = VOL_NORMALE * np.sqrt(h) * chocs  

# 1) Ce qui se passe apres le plus gros choc
jour_choc = int(np.argmax(np.abs(chocs[:-10])))
print(f"Le plus gros choc arrive au jour {jour_choc + 1} : {chocs[jour_choc]:+.2f}")
print("  Jour   Choc   Nervosite h   Volatilite du jour")
for t in range(jour_choc, jour_choc + 8):
    print(f"  {t + 1:4d}  {chocs[t]:+.2f}      {h[t]:.2f}          {VOL_NORMALE * np.sqrt(h[t]):.2%}")

# 2) La volatilite mesuree sur les 20 periodes de 50 jours
vols_calme = [np.std(calme[d:d + 50]) for d in range(0, NB_JOURS, 50)]
vols_volatile = [np.std(volatile[d:d + 50]) for d in range(0, NB_JOURS, 50)]
print("\nVolatilite mesuree sur 20 periodes de 50 jours :")
print(f"  Marche calme    : de {min(vols_calme):.2%} a {max(vols_calme):.2%}")
print(f"  Marche volatile : de {min(vols_volatile):.2%} a {max(vols_volatile):.2%}")

print(f"\nNervosite moyenne sur les 1000 jours : {h.mean():.2f}")
print(f"Volatilite mesuree sur les 1000 jours : calme {np.std(calme):.2%}, volatile {np.std(volatile):.2%}")