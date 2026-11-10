import numpy as np  

# Generateur de hasard, toujours le meme grace a 161
rng = np.random.default_rng(161)  

VRAIE_VOLATILITE = 0.02  # 

# 1) Fabriquer 10 jours de Bourse
chocs = rng.standard_normal(10)  
rendements = VRAIE_VOLATILITE * chocs

print("Les 10 jours simules :")
for jour, r in enumerate(rendements, start=1):
    print(f"  Jour {jour:2d} : {r:+.2%}")

# 2) Faire comme si on ne connaissait pas la verite, et la mesurer
print("\nVolatilite mesuree sur 5 periodes differentes de 10 jours :")
for essai in range(1, 6):
    dix_jours = VRAIE_VOLATILITE * rng.standard_normal(10)
    print(f"  Essai {essai} : {np.std(dix_jours):.2%}")

beaucoup = VRAIE_VOLATILITE * rng.standard_normal(1_000_000)
print(f"\nVolatilite mesuree sur 1 000 000 jours : {np.std(beaucoup):.2%}")
print(f"Vraie volatilite choisie au depart     : {VRAIE_VOLATILITE:.2%}")