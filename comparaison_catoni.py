"""
Compare la covariance empirique et la covariance de Catoni a la covariance theorique C,
pour plusieurs valeurs de delta.

Utilisation (avec le code du collegue) :
    df, C, corr, sigma = generate_synthetic_data(...)
    from comparaison_catoni import compare_estimators
    res = compare_estimators(df, C)                  # tableau de resultats
    res = compare_estimators(df, C, plot=True)       # + courbe erreur vs delta

sigma n'est pas necessaire : les volatilites sont deja dans C (diagonale = sigma^2).
"""
import numpy as np
import pandas as pd
from catoni_covariance import catoni_covariance

DELTAS = (1e-12, 1e-6, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 0.9)


def _rel_errors(S, C):
    """Erreurs relatives par rapport a C : norme de Frobenius et norme spectrale."""
    D = S - C
    return (np.linalg.norm(D) / np.linalg.norm(C),
            np.linalg.norm(D, 2) / np.linalg.norm(C, 2))


def compare_estimators(df, C, deltas=DELTAS, plot=False):
    """df : DataFrame des rendements (N dates x n actifs). C : covariance theorique (n x n).
    Renvoie un DataFrame (une ligne par delta) :
      err_F_*  : ||S_hat - C||_F / ||C||_F        (erreur globale)
      err_2_*  : ||S_hat - C||_2 / ||C||_2        (pire direction)
      gain_*_% : 100 * (1 - erreur Catoni / erreur empirique) ; > 0 = Catoni meilleur
    """
    X = df.to_numpy(dtype=float)
    C = np.asarray(C, dtype=float)
    if C.shape != (X.shape[1], X.shape[1]):
        raise ValueError(f"C doit etre {X.shape[1]}x{X.shape[1]}, recu {C.shape}")

    emp_F, emp_2 = _rel_errors(df.cov().to_numpy(), C)          # covariance empirique

    rows = []
    for d in deltas:
        cat_F, cat_2 = _rel_errors(catoni_covariance(X, delta=d), C)   # covariance de Catoni
        rows.append({"delta": d,
                     "err_F_emp": emp_F, "err_F_catoni": cat_F, "gain_F_%": 100 * (1 - cat_F / emp_F),
                     "err_2_emp": emp_2, "err_2_catoni": cat_2, "gain_2_%": 100 * (1 - cat_2 / emp_2)})
    res = pd.DataFrame(rows).set_index("delta")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.5))
        plt.semilogx(res.index, res["err_F_catoni"], "o-", label="Catoni")
        plt.axhline(emp_F, linestyle="--", color="black", label="Empirique")
        plt.xlabel("delta"); plt.ylabel("Erreur relative (Frobenius)")
        plt.title("Erreur par rapport a la covariance theorique C"); plt.legend()
        plt.show()
    return res


def show(res):
    """Affichage lisible : toutes les colonnes, delta en notation courte."""
    print(res.rename(index=lambda d: f"{d:g}").to_string(float_format=lambda x: f"{x:.3f}"))

if __name__ == "__main__":
    from synthetic_data import generate_synthetic_data   

    df, C, corr, sigma = generate_synthetic_data(n=20, N=10000, nu=5, seed=161)
    show(compare_estimators(df, C))
