# Régime GARCH(1,1) et estimateur de Tyler

**Projet PI² n°161 – Estimation robuste de la matrice de covariance**
Binôme : Anthony FEUDJIO DJOUMETIO et Mayeul BONNAUD
Encadrante : Mme Sirine LOUATI

Ce dossier contient tout le travail du binôme sur le **marché à volatilité variable dans le temps** (modèle GARCH(1,1)) et sur son estimateur robuste, le **M-estimateur de Tyler**. En suivant ce guide, vous obtiendrez exactement les mêmes résultats que ceux du rapport.

---

## Sommaire

1. [L'idée du projet en bref](#1-lidée-du-projet-en-bref)
2. [Organisation des fichiers](#2-organisation-des-fichiers)
3. [Installation](#3-installation)
4. [Lancer les scripts](#4-lancer-les-scripts)
5. [Semaine 1 : simulation du marché](#5-semaine-1--simulation-du-marché)
6. [Semaine 2 : covariance empirique et normes d'erreur](#6-semaine-2--covariance-empirique-et-normes-derreur)
7. [Semaines 3-4 : estimateur de Tyler](#7-semaines-3-4--estimateur-de-tyler)
8. [Résumé des résultats](#8-résumé-des-résultats)
9. [Reproductibilité et problèmes fréquents](#9-reproductibilité-et-problèmes-fréquents)
10. [Prochaines étapes](#10-prochaines-étapes)

---

## 1. L'idée du projet en bref

- La **matrice de covariance Σ** décrit le risque de chaque action et la façon dont les actions bougent ensemble. Elle sert à construire le portefeuille de variance minimale.
- En pratique, Σ est inconnue : on l'**estime** à partir des rendements passés. L'estimateur classique (la **covariance empirique**) est très sensible aux jours de crise.
- Nous **simulons** un marché dont nous connaissons la vraie Σ (le « corrigé »). Nous pouvons ainsi mesurer exactement l'erreur de chaque estimateur : ‖Σ̂ − Σ‖.
- Notre binôme étudie un marché où **la volatilité varie dans le temps** (périodes calmes et crises), et teste si **Tyler** fait mieux que la covariance empirique sur ce marché.

---

## 2. Organisation des fichiers

```
garch_tyler/
├── README_Garch_Tyler.md
├── semaine_1/                          Simulation du marché
│   ├── etape1_une_action.py            Une action, volatilité constante
│   ├── etape2_deux_actions.py          Deux actions liées par le marché
│   ├── etape3_risque_variable.py       La nervosité GARCH sur une action
│   ├── etape4_marche_complet.py        Le marché complet : 20 actions + Σ   ← module central
│   └── etape5_verifications.py         Vérifications du marché + graphique
├── semaine_2/                          Covariance empirique et normes d'erreur
│   ├── etape1_covariance_empirique.py  Calcul de Σ̂ et comparaison à Σ
│   ├── etape2_normes_erreur.py         Normes de Frobenius et spectrale     ← grille de notation
│   ├── etape3_validation_gaussienne.py Validation contre la formule théorique
│   └── etape4_garch_vs_gaussien.py     Erreur de l'empirique : GARCH contre gaussien
└── semaine_3_4/                        Estimateur de Tyler
    ├── etape1_tyler.py                 Implémentation et vérifications de Tyler
    └── etape2_tyler_vs_empirique.py    Comparaison Tyler contre empirique
```

Les scripts des semaines 2 et 3-4 **réutilisent** le marché de `semaine_1/etape4_marche_complet.py` et la grille de notation de `semaine_2/etape2_normes_erreur.py`. Il faut donc garder cette arborescence telle quelle.

---

## 3. Installation

**Prérequis :** Python 3.10 ou plus récent (testé avec Python 3.13 et 3.14).

**Bibliothèques :** `numpy` et `matplotlib`, rien d'autre.

### Windows (PowerShell)

```powershell
# 1. Se placer dans le dossier du dépôt
cd Robust-estimation-of-the-covariance-matrix

# 2. Créer et activer un environnement virtuel
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Installer les bibliothèques
pip install numpy matplotlib
```

Si PowerShell refuse d'activer l'environnement (« l'exécution de scripts est désactivée »), lancez une fois :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS / Linux

```bash
cd Robust-estimation-of-the-covariance-matrix
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

---

## 4. Lancer les scripts

**Règle importante :** placez-vous **dans le dossier du script** avant de le lancer. Les graphiques (`.png`) sont enregistrés dans le dossier courant.

### Tout lancer, dans l'ordre (PowerShell)

```powershell
cd garch_tyler\semaine_1
python .\etape1_une_action.py
python .\etape2_deux_actions.py
python .\etape3_risque_variable.py
python .\etape4_marche_complet.py
python .\etape5_verifications.py

cd ..\semaine_2
python .\etape1_covariance_empirique.py
python .\etape2_normes_erreur.py
python .\etape3_validation_gaussienne.py
python .\etape4_garch_vs_gaussien.py

cd ..\semaine_3_4
python .\etape1_tyler.py
python .\etape2_tyler_vs_empirique.py
```

Sous macOS / Linux, remplacez `\` par `/` et `python` par `python3`.

**Durée :** chaque script prend de moins d'une seconde à une quinzaine de secondes. Les plus longs sont `semaine_2/etape3`, `semaine_2/etape4` et `semaine_3_4/etape2`, qui font des milliers de simulations.

**Graphiques produits :**

| Fichier | Créé par | Dossier |
|---|---|---|
| `regime_garch.png` | `etape5_verifications.py` | `semaine_1/` |
| `erreur_selon_T.png` | `etape4_garch_vs_gaussien.py` | `semaine_2/` |
| `tyler_vs_empirique.png` | `etape2_tyler_vs_empirique.py` | `semaine_3_4/` |

---

## 5. Semaine 1 : simulation du marché

### Le modèle

Chaque rendement est la somme d'une **part de marché** (commune à toutes les actions) et d'une **part propre** (spécifique à l'action), multipliée par la racine de la **nervosité** h du marché :

```
rendement(t, i) = √h_t × ( b_i × VOL_MARCHE × choc_marché_t  +  propre_i × choc_propre_t,i )
```

La nervosité suit une récurrence GARCH(1,1) :

```
h_t = OMEGA + ALPHA × choc_marché_(t−1)² × h_(t−1) + BETA × h_(t−1)
```

La vraie matrice de covariance (le corrigé) se calcule directement à partir des paramètres :

```
Σ = VOL_MARCHE² × b bᵀ + diag(propre²)
```

Comme OMEGA = 1 − ALPHA − BETA, la nervosité vaut 1 en moyenne : **le marché GARCH a la même Σ que le marché calme**. Seule la difficulté d'estimation change.

### Les paramètres

| Paramètre | Valeur | Rôle | Pourquoi cette valeur |
|---|---|---|---|
| `GRAINE` | 161 | fixe le hasard (reproductibilité) | numéro du projet ; n'importe quel entier conviendrait |
| `N` | 20 | nombre d'actions (Σ est 20 × 20) | réaliste (190 covariances) et calculs rapides |
| `VOL_MARCHE` | 0,01 | volatilité quotidienne du marché | 1 % par jour, réaliste pour un indice actions |
| `b` | uniforme sur [0,5 ; 1,5] | sensibilité de l'action au marché | de défensive à très sensible, autour de 1 |
| `propre` | uniforme sur [0,01 ; 0,02] | risque spécifique quotidien | volatilités de 1,5 % à 2,2 % par jour, corrélations de 0,12 à 0,48 |
| `ALPHA` | 0,15 | réaction à un choc de la veille | valeur typique : crises marquées, kurtosis finie |
| `BETA` | 0,80 | mémoire de la nervosité | persistance de 0,95 : un choc s'efface de moitié en environ 13 jours |
| `OMEGA` | 1 − ALPHA − BETA = 0,05 | rappel vers la normale | nervosité moyenne égale à 1, donc même Σ |
| `h[0]` | 1 | point de départ de la nervosité | valeur moyenne, mais arbitraire |
| `jours_jetes` | 500 | jours supprimés au début | efface l'effet de `h[0]` (0,95⁵⁰⁰ ≈ 0) |

**Contraintes respectées :** ALPHA + BETA = 0,95 < 1 (nervosité stable) et BETA² + 2·ALPHA·BETA + 3·ALPHA² = 0,9475 < 1 (kurtosis finie).

### Les étapes

| Script | Ce qu'il fait | Résultat clé attendu |
|---|---|---|
| `etape1_une_action.py` | simule une action à volatilité constante de 2 % et la mesure sur des périodes de 10 jours | volatilité mesurée sur 1 000 000 jours : **2,00 %** |
| `etape2_deux_actions.py` | deux actions (Danone, Air France) liées par le marché ; corrélation mesurée contre corrélation vraie | **+0,267** mesurée contre **+0,268** vraie |
| `etape3_risque_variable.py` | ajoute la nervosité GARCH à une action | volatilité sur 50 jours : de 1,65 % à 2,34 % (calme) contre **1,32 % à 2,91 %** (volatile) |
| `etape4_marche_complet.py` | le marché complet de 20 actions et son Σ | volatilités de **1,48 % à 2,18 %**, corrélations de **0,12 à 0,48**, gros mouvements (> 3 %) : **23 %** les jours nerveux contre **3 %** les jours calmes |
| `etape5_verifications.py` | vérifie le marché et trace `regime_garch.png` | voir ci-dessous |

### Sortie attendue de `etape5_verifications.py`

```
=== Verification 1 : Sigma ===
Symetrique : True
Plus petite valeur propre : 1.305e-04 (doit etre > 0)

=== Verification 2 : les donnees visent le bon corrige ===
Nervosite moyenne : 0.989 (doit etre proche de 1)
Ecart entre la covariance mesuree et Sigma : 1.38% (doit etre petit)

=== Verification 3 : comportement de l'action 1 ===
                                  Marche calme   Marche volatile
Autocorrelation des rendements   : +0.001         +0.004
Autocorrelation des rendements^2 : +0.000         +0.189
Kurtosis                         : 3.00           4.75

Graphique enregistre : regime_garch.png
```

**Interprétation :**
- Σ est une matrice de covariance valide (symétrique, valeurs propres positives).
- Sur 500 000 jours, la covariance mesurée retombe sur Σ (1,38 % d'écart) : **Σ est bien la vraie covariance des données**, même avec le GARCH.
- Le marché volatile reproduit les marchés réels : on ne peut pas prédire la direction du lendemain (autocorrélation ≈ 0), mais les grosses variations se regroupent (autocorrélation des r² = 0,19) et les jours extrêmes sont plus fréquents (kurtosis 4,75 > 3).

<details>
<summary>Sortie complète de <code>etape4_marche_complet.py</code></summary>

```
--- Le corrige Sigma ---
Taille : (20, 20)
Volatilites : 1.48% a 2.18%
Correlations : 0.12 a 0.48

--- Les donnees simulees ---
Taille du tableau de rendements : (1000, 20)

10 premiers jours, 4 premieres actions :
  Jour  Nervosite   Action 1  Action 2  Action 3  Action 4
     1    0.66       +1.46%   +0.45%   -0.52%   -4.44%
     2    0.58       -0.70%   +0.22%   +0.99%   -0.35%
     3    0.58       +0.09%   +0.95%   -0.24%   -1.04%
     4    0.58       -0.22%   +0.64%   -0.04%   -1.17%
     5    0.58       +0.37%   +1.73%   -1.34%   +1.81%
     6    0.71       +0.56%   +0.21%   +0.92%   -2.38%
     7    0.73       -0.92%   -2.17%   -1.46%   -0.41%
     8    0.64       +0.45%   +0.18%   +3.00%   -0.49%
     9    0.71       +0.98%   +1.33%   -1.38%   +1.41%
    10    0.74       -0.69%   -1.57%   +1.38%   -0.80%

Jours nerveux (20 % les plus nerveux, h > 1.35) : 200 jours
  Mouvements de plus de 3 % : 23%
Jours calmes (20 % les plus calmes, h < 0.57) : 200 jours
  Mouvements de plus de 3 % : 3%
```
</details>

---

## 6. Semaine 2 : covariance empirique et normes d'erreur

### Les outils

- **Covariance empirique** : `np.cov(rendements, rowvar=False)`, une matrice 20 × 20 (`rowvar=False` : les colonnes sont les actions).
- **Erreur** : la matrice Σ̂ − Σ, résumée en un seul nombre par deux normes, en pourcentage de la taille de Σ :
  - **Frobenius** : ‖A‖_F = √(Σᵢ Σⱼ aᵢⱼ²), l'erreur **globale** sur toutes les cases ;
  - **spectrale** : la plus grande valeur propre de A en valeur absolue, l'erreur sur le risque du **pire portefeuille** possible.

La fonction `erreurs_relatives(sigma_chapeau, sigma)` de `etape2_normes_erreur.py` calcule les deux. C'est la **grille de notation** de tout le projet.

### Les étapes

| Script | Ce qu'il fait | Résultat clé attendu |
|---|---|---|
| `etape1_covariance_empirique.py` | calcule Σ̂ sur 3 années (250 jours) de marché gaussien et compare 3 cases à Σ | variance de l'action 2 : vraie 4,13, estimée **3,38 / 4,50 / 4,61** (× 10⁻⁴) |
| `etape2_normes_erreur.py` | vérifie les normes à la main sur un exemple 2 × 2, puis note les 3 années | Frobenius de **14 % à 17 %**, spectrale de **8 % à 13 %** |
| `etape3_validation_gaussienne.py` | compare l'erreur mesurée (500 échantillons) à la formule théorique exacte | écart simulation / théorie **≤ 1 %** |
| `etape4_garch_vs_gaussien.py` | erreur de l'empirique sur le marché gaussien contre le marché GARCH, trace `erreur_selon_T.png` | l'erreur est environ **2 fois plus grande** avec le GARCH |

### Sortie attendue de `etape3_validation_gaussienne.py`

La formule théorique (marché gaussien) est : E‖Σ̂ − Σ‖²_F = (tr(Σ)² + ‖Σ‖²_F) / (T − 1).

```
    T    Frobenius simule   Frobenius theorie   Ecart   Spectrale simulee
   50         39.8%              40.0%        -0.6%        30.2%
  100         27.9%              28.2%        -1.0%        21.1%
  250         17.8%              17.8%        +0.1%        13.4%
  500         12.6%              12.5%        +0.6%         9.6%
 1000          8.9%               8.9%        +0.6%         6.8%
```

**Interprétation :** la simulation retrouve la théorie à ±1 % : le simulateur, `np.cov` et les normes sont corrects. L'erreur diminue en 1/√T (4 fois plus de jours divisent l'erreur par 2).

### Sortie attendue de `etape4_garch_vs_gaussien.py`

```
            Frobenius               Spectrale
    T   Gaussien   GARCH   Ratio   Gaussien   GARCH   Ratio
   50     39.3%   66.7%   x1.70     30.2%   61.9%   x2.05
  100     27.6%   46.6%   x1.69     21.1%   43.1%   x2.05
  250     17.6%   35.2%   x2.00     13.4%   33.2%   x2.48
  500     12.5%   26.0%   x2.08      9.6%   24.6%   x2.55
 1000      8.8%   18.7%   x2.12      6.8%   17.6%   x2.60

1000 jours de marche volatile valent environ 223 jours de marche gaussien
```

**Interprétation :** sur le marché GARCH, la covariance empirique se trompe environ 2 fois plus, car quelques jours de crise (aux rendements énormes) dominent le calcul. Le chiffre de 223 jours suppose une erreur en 1/√T : c'est un ordre de grandeur.

---

## 7. Semaines 3-4 : estimateur de Tyler

### Le principe

La covariance empirique laisse les jours de crise dominer. Tyler **ramène d'abord chaque jour à la même taille** (sa distance de Mahalanobis x_tᵀ Σ⁻¹ x_t), puis fait la moyenne :

```
Σ = (n / T) × Σ_t  x_t x_tᵀ / (x_tᵀ Σ⁻¹ x_t)
```

Σ apparaît des deux côtés : on la calcule par **itérations successives** (point fixe). On part de la matrice identité, on applique la formule, on fixe la trace à n, et on recommence jusqu'à ce que le changement relatif soit inférieur à 10⁻⁶.

**Limite :** Tyler n'estime que la **forme** de Σ (corrélations et volatilités relatives), pas sa **taille**. C'est suffisant pour le projet, car les poids du portefeuille de variance minimale ne dépendent que de la forme de Σ. Les comparaisons portent donc sur la forme : chaque matrice est ramenée à une trace égale à n avant d'être notée.

### Les fonctions de `etape1_tyler.py`

| Fonction | Rôle |
|---|---|
| `tyler_forme(rendements)` | l'algorithme de Tyler ; renvoie la forme (trace = n) et le nombre d'itérations |
| `tyler(rendements)` | la forme de Tyler remise à la taille de la covariance empirique |
| `forme_de(sigma)` | ramène une matrice à trace = n, pour comparer des formes entre elles |

### Sortie attendue de `etape1_tyler.py`

```
=== Verification 1 : l'algorithme (250 jours GARCH) ===
Converge en 11 iterations
Symetrique : True
Trace : 20.00 (doit valoir 20)
Plus petite valeur propre : 0.267 (doit etre > 0)

=== Verification 2 : 100 000 jours GARCH ===
Erreur sur la forme : Frobenius 0.89%, spectrale 0.63% (doit etre petit)

=== Apercu : 250 jours GARCH, 3 echantillons ===
                Erreur sur la forme           Erreur complete (Frobenius)
Echantillon   Empirique   Tyler              Empirique   Tyler
     1          18.7%     14.2%              17.6%     14.7%
     2          23.7%     19.8%              85.5%     81.4%
     3          20.3%     22.1%              44.9%     47.2%
```

**Interprétation :** l'algorithme converge et, avec beaucoup de données, retrouve la forme de Σ à 0,9 % près. L'erreur complète peut être très grande (85 %) quand la nervosité moyenne de l'échantillon s'éloigne de 1 : c'est une erreur de **taille**, que Tyler ne corrige pas.

### Sortie attendue de `etape2_tyler_vs_empirique.py`

Gain = de combien Tyler réduit l'erreur de l'empirique (négatif : Tyler est moins bon).

```
--- Marche Gaussien ---
              Frobenius                      Spectrale
    T   Empirique   Tyler    Gain     Empirique   Tyler    Gain
   50      37.4%    40.3%     -8%        26.3%    28.5%     -8%
  100      26.2%    27.7%     -6%        18.4%    19.5%     -6%
  250      16.7%    17.5%     -5%        11.7%    12.3%     -6%
  500      11.8%    12.4%     -5%         8.4%     8.8%     -6%
 1000       8.3%     8.8%     -5%         5.9%     6.2%     -5%

--- Marche GARCH ---
              Frobenius                      Spectrale
    T   Empirique   Tyler    Gain     Empirique   Tyler    Gain
   50      39.6%    40.5%     -2%        27.5%    29.0%     -6%
  100      28.6%    27.9%     +3%        19.5%    19.6%     -0%
  250      19.2%    17.5%     +9%        13.3%    12.4%     +7%
  500      14.1%    12.3%    +13%         9.9%     8.6%    +13%
 1000      10.1%     8.7%    +14%         7.0%     6.1%    +14%
```

**Interprétation :**
- **Marché gaussien :** Tyler est un peu moins précis (−5 à −8 %). C'est attendu : sans jours de crise, la covariance empirique est optimale.
- **Marché GARCH :** Tyler devient meilleur dès 100 jours, jusqu'à **+14 %** pour T = 1000.
- Tyler fait **quasiment la même erreur sur les deux marchés** : il neutralise l'effet de la volatilité variable.
- Avec 50 jours pour 20 actions, les données sont trop peu nombreuses pour que Tyler gagne.

---

## 8. Résumé des résultats

| Question | Réponse |
|---|---|
| Le marché simulé est-il correct ? | Oui : la covariance mesurée sur 500 000 jours retombe sur Σ (1,38 % d'écart). |
| Le marché est-il réaliste ? | Oui : regroupement de volatilité (autocorrélation des r² = 0,19) et queues épaisses (kurtosis 4,75). |
| Le code de mesure est-il fiable ? | Oui : il retrouve la formule théorique à ±1 % sur le marché gaussien. |
| Le GARCH gêne-t-il la covariance empirique ? | Oui : son erreur est environ 2 fois plus grande (×2,1 en Frobenius, ×2,6 en spectrale pour T = 1000). |
| Tyler fait-il mieux ? | Sur la forme de Σ et sur le marché GARCH, oui : jusqu'à 14 % d'erreur en moins. Sur le marché gaussien, il est légèrement moins bon. |

---

## 9. Reproductibilité et problèmes fréquents

**Pourquoi les résultats sont identiques chez tout le monde :**
- tous les tirages aléatoires utilisent `np.random.default_rng(161)` ;
- les mêmes 20 actions sont recréées dans chaque script par `creer_actions(N)` ;
- les scripts tirent les nombres aléatoires toujours dans le même ordre.

Les résultats ont été vérifiés identiques avec Python 3.13 / NumPy 2.5 (Linux) et Python 3.14 (Windows). Avec d'autres versions, un dernier chiffre peut exceptionnellement différer, sans changer les conclusions.

**Ne modifiez pas les paramètres** (`GRAINE`, `N`, `ALPHA`, `BETA`, `NB_REPETITIONS`, `TAILLES_T`…) si vous voulez retrouver les chiffres du rapport. Si vous changez ALPHA ou BETA, recalculez toujours `OMEGA = 1 - ALPHA - BETA`, sinon Σ n'est plus le bon corrigé.

**Problèmes fréquents :**

| Problème | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'` | activez l'environnement virtuel, puis `pip install numpy matplotlib` |
| `ModuleNotFoundError: No module named 'etape4_marche_complet'` | l'arborescence a été modifiée : les dossiers `semaine_1`, `semaine_2` et `semaine_3_4` doivent être côte à côte |
| le graphique `.png` n'apparaît pas où prévu | il est enregistré dans le dossier **courant** : placez-vous dans le dossier du script avant de le lancer |
| aucune fenêtre de graphique ne s'ouvre | c'est voulu (`matplotlib.use("Agg")`) : ouvrez le fichier `.png` créé |
| PowerShell refuse `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## 10. Prochaines étapes

- **Robustesse :** tester un GARCH plus agité (ALPHA = 0,20, BETA = 0,75, kurtosis théorique d'environ 17) pour vérifier que le gain de Tyler augmente avec l'intensité des crises.
- **Comparaison croisée :** appliquer Tyler aux marchés des deux autres binômes (gaussien et Student-t) et leurs estimateurs (Median-of-Means et Catoni) à notre marché GARCH.
- **Partie 2 :** passer aux données réelles et évaluer le portefeuille de variance minimale en backtest.