import numpy as np  

rng = np.random.default_rng(161)  

# --- La verite qu'on choisit ---
VOL_MARCHE = 0.01 
B_DANONE, B_AIRFRANCE = 0.5, 1.5
PROPRE_DANONE, PROPRE_AIRFRANCE = 0.01, 0.02 


def simuler(nb_jours):
    """Fabrique nb_jours de rendements pour Danone et Air France."""
    choc_marche = rng.standard_normal(nb_jours)
    choc_danone = rng.standard_normal(nb_jours)  
    choc_airfrance = rng.standard_normal(nb_jours)

    # Rendement = part venant du marche + part propre a l'entreprise
    danone = B_DANONE * VOL_MARCHE * choc_marche + PROPRE_DANONE * choc_danone
    airfrance = B_AIRFRANCE * VOL_MARCHE * choc_marche + PROPRE_AIRFRANCE * choc_airfrance
    return danone, airfrance


# 1) Regarder 5 jours
danone, airfrance = simuler(5)
print("5 jours simules :")
print("  Jour   Danone   Air France")
for jour in range(5):
    print(f"  {jour + 1:4d}  {danone[jour]:+7.2%}   {airfrance[jour]:+7.2%}")

# 2) La vraie correlation, calculee a partir de nos choix
cov_vraie = B_DANONE * B_AIRFRANCE * VOL_MARCHE**2
vol_vraie_d = np.sqrt((B_DANONE * VOL_MARCHE) ** 2 + PROPRE_DANONE**2)
vol_vraie_a = np.sqrt((B_AIRFRANCE * VOL_MARCHE) ** 2 + PROPRE_AIRFRANCE**2)
corr_vraie = cov_vraie / (vol_vraie_d * vol_vraie_a)

# 3) La mesurer a partir des donnees seules
print("\nCorrelation mesuree sur 5 periodes differentes de 50 jours :")
for essai in range(1, 6):
    d, a = simuler(50)
    print(f"  Essai {essai} : {np.corrcoef(d, a)[0, 1]:+.2f}")

d, a = simuler(1_000_000)
print(f"\nCorrelation mesuree sur 1 000 000 jours : {np.corrcoef(d, a)[0, 1]:+.3f}")
print(f"Vraie correlation choisie au depart     : {corr_vraie:+.3f}")