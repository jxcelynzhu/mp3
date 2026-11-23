import numpy as np


def sample_mean_cov(X):
    """Return the sample mean and unbiased covariance matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be a 2D array with at least two samples")
    mean = np.mean(X, axis=0)
    centered = X - mean
    covariance = (centered.T @ centered) / (X.shape[0] - 1)
    return mean, covariance


def is_valid_covariance(S, tol=1e-10):
    """Check if S is symmetric and positive semidefinite."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or not np.all(np.isfinite(S)):
        return False
    if not np.allclose(S, S.T, atol=tol, rtol=0):
        return False
    eigenvalues = np.linalg.eigvalsh((S + S.T) / 2)
    return bool(np.min(eigenvalues) >= -tol)


def gaussian_pdf(x, mu, Sigma):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    d = mu.shape[0]
    det = np.linalg.det(Sigma)
    if det <= 0:
        raise ValueError("Covariance matrix must be positive definite (det > 0).")

    diff = x - mu
    orig_shape = diff.shape[:-1]
    diff_2d = diff.reshape(-1, d)

    solved = np.linalg.solve(Sigma, diff_2d.T)
    maha_sq = np.sum(diff_2d.T * solved, axis=0)

    norm_const = 1.0 / np.sqrt(((2.0 * np.pi) ** d) * det)
    pdf = norm_const * np.exp(-0.5 * maha_sq)

    return pdf.reshape(orig_shape)


def whitening_matrix(Sigma):
    Sigma = np.asarray(Sigma, dtype=float)
    eigvals, V = np.linalg.eigh(Sigma)
    if np.any(eigvals <= 0):
        raise ValueError("Covariance matrix must be positive definite to whiten.")

    W = (1.0 / np.sqrt(eigvals))[:, None] * V.T
    return W


def lmmse_weights(Sigma_xx, Sigma_sx):
    return np.linalg.solve(Sigma_xx, np.asarray(Sigma_sx).reshape(-1))
