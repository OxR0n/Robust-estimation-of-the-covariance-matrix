import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, norm


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


def compare_matrices(A, B, title, cmap, vmin=None, vmax=None):
    if vmin is None:
        vmin = min(A.min(), B.min())
    if vmax is None:
        vmax = max(A.max(), B.max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    im = axes[0].imshow(A, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('Theoretical ' + title)
    axes[1].imshow(B, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title('Empirical ' + title)
    fig.colorbar(im, ax=axes)
    plt.show()


if __name__ == '__main__':
    # Simulation parameters (None generates random volatilities/correlations)
    n = 10
    N = 1000
    nu = 5
    sigma = None
    corr = None
    mu = None

    # Example for n=3: corr = [0.6, 0.3, 0.4]
    df, C, corr, sigma = generate_synthetic_data(
        n, N, nu, sigma, corr, mu, seed=42
    )

    C_empirical = df.cov().values
    corr_empirical = df.corr().values
    error = np.linalg.norm(C_empirical - C, ord='fro')
    relative_error = error / np.linalg.norm(C, ord='fro')

    print('Volatilities:', sigma)
    print('\nTheoretical correlation matrix:\n', corr)
    print('\nTheoretical covariance matrix:\n', C)
    print('\nEmpirical covariance matrix:\n', C_empirical)
    print('\nAbsolute error:', error)
    print('Relative error:', relative_error)
    print('\nSynthetic data:\n', df.head())

    compare_matrices(corr, corr_empirical, 'correlation matrix', 'coolwarm', -1, 1)
    compare_matrices(C, C_empirical, 'covariance matrix', 'viridis')

    plt.figure(figsize=(12, 4))
    plt.plot(df['Asset_1'], linewidth=0.8)
    plt.title('Synthetic Asset Returns')
    plt.xlabel('Observation')
    plt.ylabel('Return')
    plt.show()

    x = np.linspace(-7, 7, 2000)
    s = np.sqrt((nu - 2) / nu)
    for log_scale in (False, True):
        plt.figure(figsize=(10, 5))
        plt.plot(x, norm.pdf(x), label='Normal distribution')
        plt.plot(x, t.pdf(x, df=nu, scale=s), label=f'Student (nu={nu})')
        if log_scale:
            plt.yscale('log')
        plt.title('Normal vs Student Distribution')
        plt.xlabel('Standardised returns')
        plt.ylabel('Density' if not log_scale else 'Density (log scale)')
        plt.legend()
        plt.show()
