"""Core evaluation metrics for multimodal reliance-utility misalignment.

Pure NumPy implementations of:
- Jensen-Shannon divergence (JSD)
- Modality utility and reliance curves
- Moving average smoothing
- Utility crossing point S*
- Normalized misalignment index (MI)
- Bootstrap confidence interval estimation with non-crossing tracking
"""

import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np


class BootstrapResult(tuple):
    """Named 3-tuple (point_estimate, lo, hi) preserving 3-item unpacking while tracking resample failures."""

    def __new__(
        cls,
        point_estimate: Optional[float],
        lo: Optional[float],
        hi: Optional[float],
        no_crossing_fraction: float = 0.0,
    ):
        return super().__new__(cls, (point_estimate, lo, hi))

    def __init__(
        self,
        point_estimate: Optional[float],
        lo: Optional[float],
        hi: Optional[float],
        no_crossing_fraction: float = 0.0,
    ):
        self.point_estimate = point_estimate
        self.lo = lo
        self.hi = hi
        self.no_crossing_fraction = float(no_crossing_fraction)

    def __repr__(self) -> str:
        return (
            f"BootstrapResult(point_estimate={self.point_estimate}, "
            f"lo={self.lo}, hi={self.hi}, "
            f"no_crossing_fraction={self.no_crossing_fraction:.4f})"
        )


def jsd(
    p: Union[Sequence[float], np.ndarray],
    q: Union[Sequence[float], np.ndarray],
    eps: float = 1e-12,
) -> float:
    """Compute Jensen-Shannon divergence in log base 2, bounded in [0, 1].

    Supports 1D probability vectors as well as 2D batches (returns the mean over the batch).
    Applies epsilon smoothing for numerical stability and zero-support probabilities.

    Parameters
    ----------
    p : array-like
        First probability distribution or batch of distributions (shape (K,) or (B, K)).
    q : array-like
        Second probability distribution or batch of distributions (shape (K,) or (B, K)).
    eps : float, default=1e-12
        Epsilon smoothing parameter added to distributions before normalization.

    Returns
    -------
    float
        Jensen-Shannon divergence value in [0, 1].
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    if p_arr.shape != q_arr.shape and not (p_arr.ndim == 1 and q_arr.ndim == 1):
        # Allow broadcasting if compatible, else check matching shapes
        p_arr, q_arr = np.broadcast_arrays(p_arr, q_arr)

    # Ensure non-negative
    p_arr = np.maximum(p_arr, 0.0)
    q_arr = np.maximum(q_arr, 0.0)

    # Epsilon smoothing
    p_smooth = p_arr + eps
    q_smooth = q_arr + eps

    # Normalize along the last axis to sum to 1
    p_norm = p_smooth / np.sum(p_smooth, axis=-1, keepdims=True)
    q_norm = q_smooth / np.sum(q_smooth, axis=-1, keepdims=True)

    # Check identical distributions shortcut
    if np.allclose(p_arr, q_arr, atol=1e-12):
        return 0.0

    # Midpoint distribution
    m = 0.5 * (p_norm + q_norm)

    # KL divergences with log base 2: D_KL(P || M) = sum(P * log2(P / M))
    kl_pm = np.sum(p_norm * np.log2(p_norm / m), axis=-1)
    kl_qm = np.sum(q_norm * np.log2(q_norm / m), axis=-1)

    div = 0.5 * (kl_pm + kl_qm)

    # Mean over batch if multi-dimensional
    if div.ndim > 0:
        mean_div = float(np.mean(div))
    else:
        mean_div = float(div)

    # Enforce theoretical bounds [0, 1]
    return float(np.clip(mean_div, 0.0, 1.0))


def utility(
    perf_degraded: Union[Sequence[float], np.ndarray],
    perf_null: Union[Sequence[float], np.ndarray, float],
) -> np.ndarray:
    """Compute modality utility curve U(S) = perf_degraded(S) - perf_null.

    Parameters
    ----------
    perf_degraded : array-like
        Performance values over a severity grid S.
    perf_null : array-like or float
        Performance under null/unimodal modality ablation baseline.

    Returns
    -------
    np.ndarray
        Utility array U(S) of same length as perf_degraded.
    """
    return np.asarray(perf_degraded, dtype=float) - np.asarray(perf_null, dtype=float)


def reliance(divergences_over_grid: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Construct reliance curve R(S) from divergence measurements over a severity grid.

    Parameters
    ----------
    divergences_over_grid : array-like
        Array of divergence values (e.g. JSD between predictions under degradation S and null).

    Returns
    -------
    np.ndarray
        Reliance array R(S).
    """
    return np.asarray(divergences_over_grid, dtype=float)


def normalized_reliance(R: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Compute normalized reliance R_hat(S) = R(S) / R(0).

    Parameters
    ----------
    R : array-like
        Reliance curve array over severity grid.

    Returns
    -------
    np.ndarray
        Normalized reliance array R_hat(S).

    Raises
    ------
    ValueError
        If R is empty or if R[0] is zero or near-zero.
    """
    R_arr = np.asarray(R, dtype=float)
    if R_arr.size == 0:
        raise ValueError("Reliance array R cannot be empty.")
    if np.isclose(R_arr[0], 0.0, atol=1e-12):
        raise ValueError(
            f"Initial reliance R[0] is zero or near-zero ({R_arr[0]}); cannot normalize reliance."
        )
    return R_arr / R_arr[0]


def smooth(x: Union[Sequence[float], np.ndarray], window: int = 3) -> np.ndarray:
    """Compute moving average smoothing of same length as input.

    Uses an edge-adjusted moving window to preserve signal magnitude at boundaries
    without attenuation artifacts.

    Parameters
    ----------
    x : array-like
        Input 1D array to smooth.
    window : int, default=3
        Moving average window size. If window <= 1, returns a copy of x.

    Returns
    -------
    np.ndarray
        Smoothed array of identical length.
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("smooth expects a 1D sequence or array.")
    if window <= 1 or len(x_arr) <= 1:
        return x_arr.copy()

    w = int(window)
    kernel = np.ones(w, dtype=float)
    sums = np.convolve(x_arr, kernel, mode="same")
    counts = np.convolve(np.ones_like(x_arr, dtype=float), kernel, mode="same")
    return sums / counts


def utility_crossing(
    U: Union[Sequence[float], np.ndarray],
    S_grid: Union[Sequence[float], np.ndarray],
    window: int = 3,
) -> Optional[float]:
    """Find the empirical utility crossing severity S* where smoothed U(S) <= 0.

    Parameters
    ----------
    U : array-like
        Utility curve values over severity grid.
    S_grid : array-like
        Severity grid values corresponding to U.
    window : int, default=3
        Window size used to smooth U before locating the zero-crossing.

    Returns
    -------
    Optional[float]
        First severity value S* where smoothed U(S) <= 0, or None if no crossing occurs.
    """
    U_arr = np.asarray(U, dtype=float)
    S_arr = np.asarray(S_grid, dtype=float)

    if len(U_arr) != len(S_arr):
        raise ValueError(
            f"Length mismatch: U has length {len(U_arr)}, S_grid has length {len(S_arr)}."
        )
    if len(U_arr) == 0:
        return None

    U_smooth = smooth(U_arr, window=window) if window > 1 else U_arr
    crossing_indices = np.where(U_smooth <= 0.0)[0]

    if len(crossing_indices) == 0:
        return None

    return float(S_arr[crossing_indices[0]])


def misalignment_index(
    R_hat: Union[Sequence[float], np.ndarray],
    S_grid: Union[Sequence[float], np.ndarray],
    S_star: Optional[float],
) -> Optional[float]:
    """Calculate Normalized Misalignment Index (MI).

    MI = (1 / (1 - S*)) * integral_{S*}^1 R_hat(S) dS using the trapezoid rule.

    MI is NOT clipped at 1.0. Values exceeding 1.0 occur when reliance grows
    past its clean-data baseline (R_hat(S) > 1 for S >= S*), indicating severe
    and increasing dependence on a degraded modality. Callers should not assume
    MI in [0, 1].

    Parameters
    ----------
    R_hat : array-like
        Normalized reliance values over severity grid S_grid.  Since R_hat =
        R(S) / R(0) and R(S) >= 0, values are non-negative but may exceed 1.
    S_grid : array-like
        Severity grid values (strictly increasing from 0 to 1).
    S_star : Optional[float]
        Utility crossing point S*.  If None, or if S_star equals the last grid
        point (no usable integration interval remains), returns None.

    Returns
    -------
    Optional[float]
        Misalignment index (>= 0, possibly > 1), or None if S_star is None
        or if S_star is at or beyond the last grid point.
    """
    if S_star is None:
        return None

    R_arr = np.asarray(R_hat, dtype=float)
    S_arr = np.asarray(S_grid, dtype=float)

    if len(R_arr) != len(S_arr):
        raise ValueError(
            f"Length mismatch: R_hat has length {len(R_arr)}, S_grid has length {len(S_arr)}."
        )

    # Edge case: crossing point at or beyond the last grid point — no usable
    # integration interval; treat as no crossing.
    last_grid = float(S_arr[-1])
    if S_star >= last_grid or np.isclose(S_star, last_grid, atol=1e-12):
        return None

    # Build evaluation segment from S* to end of grid
    # Interpolate R_hat at S* if S* does not lie exactly on a grid point
    mask_after = S_arr > S_star
    sub_S = S_arr[mask_after]
    sub_R = R_arr[mask_after]

    if np.any(np.isclose(S_arr, S_star, atol=1e-12)):
        # S* is on the grid
        mask_geq = S_arr >= (S_star - 1e-12)
        eval_S = S_arr[mask_geq]
        eval_R = R_arr[mask_geq]
    else:
        # Interpolate at S* and prepend
        r_star_interp = float(np.interp(S_star, S_arr, R_arr))
        eval_S = np.concatenate(([S_star], sub_S))
        eval_R = np.concatenate(([r_star_interp], sub_R))

    if len(eval_S) < 2:
        # Only one point — return its value unclipped (R_hat >= 0)
        return float(eval_R[0])

    # Numerical integration via trapezoid rule (no upper clip)
    integral = float(np.sum(0.5 * (eval_R[:-1] + eval_R[1:]) * np.diff(eval_S)))
    normalization_factor = 1.0 / (1.0 - S_star)
    mi = normalization_factor * integral

    return float(mi)


def bootstrap_ci(
    metric_fn: Callable[[Any], Optional[float]],
    per_item_data: Sequence[Any],
    n_boot: int = 1000,
    seed: int = 0,
) -> BootstrapResult:
    """Compute 95% bootstrap confidence intervals with non-crossing failure tracking.

    Resamples items with replacement. If `metric_fn` returns None on a resample
    (e.g., when no utility crossing S* exists), the resample is tracked and the
    fraction of non-crossing resamples is reported.

    Parameters
    ----------
    metric_fn : Callable[[Sequence[Any]], Optional[float]]
        Function computing the target metric from a sample/batch of item-level data.
    per_item_data : Sequence[Any]
        Sequence or array of per-item observations to resample.
    n_boot : int, default=1000
        Number of bootstrap iterations.
    seed : int, default=0
        Random seed for the bootstrap generator.

    Returns
    -------
    BootstrapResult
        A 3-tuple (point_estimate, lo, hi) at 95% CI with attribute `no_crossing_fraction`.
    """
    point_estimate = metric_fn(per_item_data)
    n_items = len(per_item_data)
    if n_items == 0:
        return BootstrapResult(point_estimate, None, None, no_crossing_fraction=0.0)

    rng = np.random.default_rng(seed)
    boot_estimates: List[float] = []
    none_count = 0

    is_ndarray = isinstance(per_item_data, np.ndarray)

    for _ in range(n_boot):
        sample_indices = rng.integers(0, n_items, size=n_items)
        if is_ndarray:
            resample = per_item_data[sample_indices]
        else:
            resample = [per_item_data[i] for i in sample_indices]

        val = metric_fn(resample)
        if val is None:
            none_count += 1
        else:
            boot_estimates.append(float(val))

    no_crossing_fraction = none_count / n_boot

    if none_count > 0:
        warnings.warn(
            f"Bootstrap resampling encountered {none_count}/{n_boot} ({no_crossing_fraction:.2%}) "
            "resamples with no crossing / None result.",
            UserWarning,
            stacklevel=2,
        )

    if len(boot_estimates) == 0:
        lo, hi = None, None
    else:
        lo = float(np.percentile(boot_estimates, 2.5))
        hi = float(np.percentile(boot_estimates, 97.5))

    return BootstrapResult(point_estimate, lo, hi, no_crossing_fraction=no_crossing_fraction)
