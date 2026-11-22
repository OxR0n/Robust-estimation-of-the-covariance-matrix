import numpy as np  

# Generateur de hasard, ca c'est pour garder la meme sequence comme cela si on re-execute le program on aura les resultats similaires
rng = np.random.default_rng(161)  

VRAIE_VOLATILITE = 0.02  # 

# On genere 10 jours de rendements ici qui est notre petite bourse.
chocs = rng.standard_normal(10)  
rendements = VRAIE_VOLATILITE * chocs

print("Les 10 jours simules :")
for jour, r in enumerate(rendements, start=1):
    print(f"  Jour {jour:2d} : {r:+.2%}")

# On calcule la volatilite sur des periodes differentes pour confirmer si ca tients la route avec ce qu'on a declarer en haut
print("\nVolatilite mesuree sur 5 periodes differentes de 10 jours :")
for essai in range(1, 21):
    dix_jours = VRAIE_VOLATILITE * rng.standard_normal(10)
    print(f"  Essai {essai} : {np.std(dix_jours):.2%}")

beaucoup = VRAIE_VOLATILITE * rng.standard_normal(1_000_000)
print(f"\nVolatilite mesuree sur 1 000 000 jours : {np.std(beaucoup):.2%}")
print(f"Vraie volatilite choisie au depart     : {VRAIE_VOLATILITE:.2%}")