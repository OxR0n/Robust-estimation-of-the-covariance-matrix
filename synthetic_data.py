import numpy as np
import pandas as pd


def random_correlation(n, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    A = rng.normal(size=(n, n))
    B = A @ A.T
    d = np.sqrt(np.diag(B))
    return B / np.outer(d, d)


def generate_synthetic_data(n, N, nu=5, sigma=None, corr=None,
                            mu=None, seed=None):
    if not np.isfinite(nu) or nu <= 2:
        raise ValueError('nu must be greater than 2')
    if n < 1 or N < 2:
        raise ValueError('Invalid number of assets or observations')

    rng = np.random.default_rng(seed)

    mu = np.zeros(n) if mu is None else np.asarray(mu, dtype=float)
    sigma = (rng.uniform(0.01, 0.03, size=n) if sigma is None
             else np.asarray(sigma, dtype=float))

    if sigma.shape != (n,) or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError('Invalid volatility vector')
    if mu.shape != (n,) or not np.all(np.isfinite(mu)):
        raise ValueError('Invalid mean vector')

    if corr is None:
        corr = random_correlation(n, rng)
    else:
        corr = np.asarray(corr, dtype=float)
        if corr.ndim == 1:
            if len(corr) != n * (n - 1) // 2:
                raise ValueError('Invalid correlation vector length')
            P = np.eye(n)
            idx = np.triu_indices(n, k=1)
            P[idx] = corr
            P[idx[1], idx[0]] = corr
            corr = P

    if corr.shape != (n, n) or not np.all(np.isfinite(corr)):
        raise ValueError('Invalid correlation matrix')
    if not np.allclose(corr, corr.T):
        raise ValueError('Correlation matrix must be symmetric')
    if not np.allclose(np.diag(corr), 1):
        raise ValueError('Diagonal elements must be equal to 1')
    if np.any(np.linalg.eigvalsh(corr) <= 0):
        raise ValueError('Correlation matrix must be positive definite')

    # Theoretical covariance matrix
    D = np.diag(sigma)
    C = D @ corr @ D

    # Correlated Gaussian returns
    L = np.linalg.cholesky(C)
    Z = rng.normal(size=(N, n))
    X = Z @ L.T

    # Common chi-square variable for all assets in each observation
    U = rng.chisquare(nu, size=N)
    T = X / np.sqrt(U[:, None] / nu)

    # Preserve the theoretical covariance matrix
    s = np.sqrt((nu - 2) / nu)
    R = mu + s * T

    df = pd.DataFrame(R, columns=[f'Asset_{i+1}' for i in range(n)])
    return df, C, corr, sigma



if __name__ == '__main__':
    df, C, corr, sigma = generate_synthetic_data(
        n=3, N=100000, nu=5, seed=42
    )
    print(df.head())
    print("\nTheoretical covariance matrix:")
    print(C)
    print("\nSample covariance (sanity check):")
    print(df.cov().values)
    assert df.shape == (100000, 3)
    assert np.all(np.isfinite(df.values))
    assert np.linalg.norm(df.cov().values - C) / np.linalg.norm(C) < 0.1
    print("\nSimulation check passed.")
