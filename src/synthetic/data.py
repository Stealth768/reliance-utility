"""Synthetic 2-modality classification benchmark data generator and degradation functions."""

from typing import NamedTuple, Optional, Tuple, Union
import numpy as np


class SyntheticData(NamedTuple):
    """Container for synthetic multimodal dataset."""

    X_A: np.ndarray
    X_B: np.ndarray
    y: np.ndarray
    mu_A: np.ndarray
    mu_B: np.ndarray


def make_data(
    n: int,
    k: int = 4,
    d: int = 8,
    seed: int = 42,
    sigma_A: float = 0.5,
    sigma_B: float = 1.8,
    mu_A: Optional[np.ndarray] = None,
    mu_B: Optional[np.ndarray] = None,
) -> SyntheticData:
    """Generate synthetic 2-modality classification data with class prototypes.

    Class y is drawn uniformly from {0, ..., k-1}.
    Prototypes mu_A, mu_B are random vectors in R^d fixed by seed (or provided).
    Clean features:
        X_A = mu_A[y] + N(0, sigma_A^2 I)
        X_B = mu_B[y] + N(0, sigma_B^2 I)

    With default sigma_A=0.5 and sigma_B=1.8, clean accuracy is ~0.95 with A alone
    and ~0.60 with B alone.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    k : int, default=4
        Number of latent classes.
    d : int, default=8
        Feature dimensionality per modality.
    seed : int, default=42
        Random seed for prototype generation (if not passed) and feature noise.
    sigma_A : float, default=0.5
        Standard deviation of modality A (strongly informative).
    sigma_B : float, default=1.8
        Standard deviation of modality B (weaker but informative).
    mu_A : Optional[np.ndarray], default=None
        Pre-existing class prototypes for modality A, shape (k, d).
    mu_B : Optional[np.ndarray], default=None
        Pre-existing class prototypes for modality B, shape (k, d).

    Returns
    -------
    SyntheticData
        Named tuple (X_A, X_B, y, mu_A, mu_B).
    """
    rng = np.random.default_rng(seed)

    # Class prototypes in R^d
    if mu_A is None:
        mu_A = rng.normal(0.0, 1.0, size=(k, d))
    if mu_B is None:
        mu_B = rng.normal(0.0, 1.0, size=(k, d))

    # Balanced class assignments
    y = rng.integers(0, k, size=n)

    # Clean features
    X_A = mu_A[y] + rng.normal(0.0, sigma_A, size=(n, d))
    X_B = mu_B[y] + rng.normal(0.0, sigma_B, size=(n, d))

    return SyntheticData(X_A=X_A, X_B=X_B, y=y, mu_A=mu_A, mu_B=mu_B)


def degrade_A(
    X_A: np.ndarray,
    S: float,
    seed: Optional[int] = None,
    tau: Optional[float] = None,
) -> np.ndarray:
    """Apply parameterized corruption to modality A.

    T(X_A, S) = (1 - S) * X_A + S * N(0, tau^2 I), where S in [0, 1].
    tau is chosen so the pure-noise end (S=1) has the same scale as clean data.

    Parameters
    ----------
    X_A : np.ndarray
        Clean or input modality A feature tensor of shape (n, d).
    S : float
        Corruption severity in [0, 1].
    seed : Optional[int], default=None
        Random seed for noise generation.
    tau : Optional[float], default=None
        Scale of the pure noise. Defaults to np.std(X_A).

    Returns
    -------
    np.ndarray
        Degraded modality A feature tensor.
    """
    S_clipped = float(np.clip(S, 0.0, 1.0))
    if S_clipped == 0.0:
        return X_A.copy()

    if tau is None:
        tau = float(np.std(X_A))

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, tau, size=X_A.shape)

    return (1.0 - S_clipped) * X_A + S_clipped * noise


def null_A(
    kind: str,
    X_A: Optional[np.ndarray] = None,
    dataset_mean: Optional[np.ndarray] = None,
    n: Optional[int] = None,
    d: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Generate null baseline input for modality A.

    Variants:
    - 'removed': None (modality A omitted from model input entirely).
    - 'zeros': all-zero tensor of same shape as X_A.
    - 'mean': constant tensor populated by dataset-level mean of X_A.

    Parameters
    ----------
    kind : str
        Null variant, must be 'removed', 'zeros', or 'mean'.
    X_A : Optional[np.ndarray], default=None
        Reference feature tensor to extract shape or mean from.
    dataset_mean : Optional[np.ndarray], default=None
        Precomputed dataset mean of X_A.
    n : Optional[int], default=None
        Number of samples (if X_A is not provided).
    d : Optional[int], default=None
        Dimensionality (if X_A is not provided).

    Returns
    -------
    Optional[np.ndarray]
        Null feature tensor for modality A, or None for kind='removed'.
    """
    if kind not in ("removed", "zeros", "mean"):
        raise ValueError(
            f"Unknown null kind '{kind}'. Expected 'removed', 'zeros', or 'mean'."
        )

    if kind == "removed":
        return None

    if X_A is not None:
        target_n, target_d = X_A.shape
    elif n is not None and d is not None:
        target_n, target_d = n, d
    else:
        raise ValueError("Must provide either X_A or both n and d.")

    if kind == "zeros":
        return np.zeros((target_n, target_d), dtype=float)

    # kind == "mean"
    if dataset_mean is None:
        if X_A is None:
            raise ValueError("Must provide dataset_mean when X_A is None for kind='mean'.")
        dataset_mean = np.mean(X_A, axis=0)

    return np.tile(dataset_mean, (target_n, 1))
