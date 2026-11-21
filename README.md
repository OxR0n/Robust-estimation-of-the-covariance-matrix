# Robust-estimation-of-the-covariance-matrix

## Régime calme gaussien

Le scénario gaussien constitue une référence contrôlée : les rendements sont
simulés indépendamment selon une loi normale multivariée dont la covariance est
connue. Il ne modélise ni volatilité variable ni changement de régime.

Les modules sont séparés pour faciliter leur intégration avec les autres
régimes :

- `simulation_gaussian.py` génère les rendements et retourne `X`, `Sigma_true`
  et les paramètres du scénario ;
- `gaussian_evaluation.py` calcule les erreurs de covariance et le risque des
  portefeuilles de variance minimale ;
- `gaussian_benchmark.py` compare la covariance empirique et Ledoit-Wolf sur
  plusieurs dimensions et tailles d’échantillon.

Installer les dépendances puis lancer un benchmark rapide :

```powershell
python -m pip install -r requirements.txt
python gaussian_benchmark.py --quick
```

Le benchmark complet effectue 100 répétitions par configuration :

```powershell
python gaussian_benchmark.py
```

Les résultats bruts, la synthèse et un graphique sont enregistrés dans
`results_gaussian/`.
Les tests peuvent être lancés avec `python -m unittest discover -s tests -v`.

Les estimateurs supplémentaires des autres binômes pourront être ajoutés au
benchmark via le paramètre `estimators` de `run_benchmark`. Les résultats de
simulation sont des mesures contrôlées, pas des résultats sur données réelles.