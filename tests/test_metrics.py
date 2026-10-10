"""Unit tests for the metrics module."""

import math
import numpy as np
import pytest

from src.metrics import (
    bootstrap_ci,
    jsd,
    misalignment_index,
    normalized_reliance,
    reliance,
    smooth,
    utility,
    utility_crossing,
)


def test_jsd_identical_distributions():
    """jsd(p, p) == 0 for various distributions."""
    p1 = np.array([0.5, 0.5])
    assert jsd(p1, p1) == 0.0

    p2 = np.array([0.2, 0.5, 0.3])
    assert jsd(p2, p2) == 0.0

    p3 = np.array([1.0, 0.0, 0.0])
    assert jsd(p3, p3) == 0.0


def test_jsd_disjoint_distributions():
    """jsd of disjoint distributions ~= 1."""
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    val = jsd(p, q)
    assert math.isclose(val, 1.0, rel_tol=1e-5, abs_tol=1e-5)

    p_multi = np.array([1.0, 0.0, 0.0, 0.0])
    q_multi = np.array([0.0, 0.0, 1.0, 0.0])
    assert math.isclose(jsd(p_multi, q_multi), 1.0, rel_tol=1e-5, abs_tol=1e-5)


def test_jsd_symmetric():
    """jsd is symmetric: jsd(p, q) == jsd(q, p)."""
    p = np.array([0.1, 0.7, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    assert math.isclose(jsd(p, q), jsd(q, p), rel_tol=1e-9, abs_tol=1e-9)


def test_jsd_bounded_in_0_1():
    """jsd is always bounded in [0, 1]."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        p = rng.dirichlet(np.ones(5))
        q = rng.dirichlet(np.ones(5))
        val = jsd(p, q)
        assert 0.0 <= val <= 1.0


def test_jsd_batch_support():
    """jsd supports 2D batches and returns mean over batch."""
    p_batch = np.array([[1.0, 0.0], [0.5, 0.5]])
    q_batch = np.array([[0.0, 1.0], [0.5, 0.5]])
    # First pair is disjoint (1.0), second is identical (0.0) -> mean ~= 0.5
    val = jsd(p_batch, q_batch)
    assert math.isclose(val, 0.5, rel_tol=1e-4, abs_tol=1e-4)


def test_utility_computation():
    """utility returns U(S) = perf_degraded - perf_null."""
    perf_deg = np.array([0.8, 0.6, 0.4, 0.2])
    perf_null = 0.3
    u = utility(perf_deg, perf_null)
    expected = np.array([0.5, 0.3, 0.1, -0.1])
    assert np.allclose(u, expected)


def test_reliance_and_normalization():
    """reliance returns R(S) and normalized_reliance raises error if R[0] is ~0."""
    divs = [0.4, 0.3, 0.2, 0.1]
    r = reliance(divs)
    r_hat = normalized_reliance(r)
    assert np.allclose(r_hat, [1.0, 0.75, 0.5, 0.25])

    # Error when R[0] ~ 0
    with pytest.raises(ValueError, match="Initial reliance R\\[0\\] is zero or near-zero"):
        normalized_reliance([0.0, 0.2, 0.3])


def test_smooth():
    """smooth preserves length and preserves constant signals."""
    x = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    s = smooth(x, window=3)
    assert len(s) == len(x)
    assert np.allclose(s, x)

    # Moving average on noisy signal
    noisy = np.array([0.0, 10.0, 0.0])
    smoothed = smooth(noisy, window=3)
    assert len(smoothed) == 3


def test_utility_crossing_always_positive():
    """utility_crossing returns None for always-positive U."""
    s_grid = np.linspace(0, 1, 11)
    u_positive = np.array([0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.01])
    assert utility_crossing(u_positive, s_grid) is None


def test_utility_crossing_hand_built():
    """utility_crossing returns right grid point for hand-built U that crosses."""
    s_grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Without smoothing (window=1) or with hand-built crossing
    u = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
    # Crossing occurs at index 3 -> 0.6
    s_star = utility_crossing(u, s_grid, window=1)
    assert s_star == 0.6


def test_misalignment_index_none_crossing():
    """misalignment_index returns None when S_star is None."""
    s_grid = np.linspace(0, 1, 6)
    r_hat = np.ones(6)
    assert misalignment_index(r_hat, s_grid, None) is None


def test_misalignment_index_extremes():
    """misalignment_index is ~0 when R_hat drops to 0 right at S*, and ~1 when R_hat stays at 1."""
    s_grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    s_star = 0.4

    # Drops to 0 right at S*
    r_hat_drops = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    mi_zero = misalignment_index(r_hat_drops, s_grid, s_star)
    assert mi_zero is not None
    assert math.isclose(mi_zero, 0.0, abs_tol=1e-5)

    # Stays at 1 after S*
    r_hat_persists = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    mi_one = misalignment_index(r_hat_persists, s_grid, s_star)
    assert mi_one is not None
    assert math.isclose(mi_one, 1.0, abs_tol=1e-5)


def test_bootstrap_ci():
    """bootstrap_ci resamples items, returns (point_estimate, lo, hi), and handles None."""
    # Data with reliable positive mean
    data = [2.0, 3.0, 4.0, 5.0, 6.0]
    pe, lo, hi = bootstrap_ci(lambda d: float(np.mean(d)), data, n_boot=200, seed=42)
    assert math.isclose(pe, 4.0, abs_tol=1e-4)
    assert lo <= pe <= hi

    # Metric function that occasionally produces None
    def flakey_crossing(items):
        mean_val = float(np.mean(items))
        return mean_val if mean_val > 0 else None

    mixed_data = [-1.0, 0.0, 0.5, 1.0]
    res = bootstrap_ci(flakey_crossing, mixed_data, n_boot=200, seed=42)
    pe, lo, hi = res
    assert hasattr(res, "no_crossing_fraction")
    assert 0.0 <= res.no_crossing_fraction <= 1.0


def test_misalignment_index_exceeds_one():
    """MI > 1 when R_hat > 1 everywhere after S* (reliance grows past clean level)."""
    s_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    s_star = 0.5
    # R_hat = 1.5 for all S >= S* → MI should be 1.5, not clipped to 1.0
    r_hat = np.array([1.0, 1.0, 1.5, 1.5, 1.5])
    mi = misalignment_index(r_hat, s_grid, s_star)
    assert mi is not None
    assert mi > 1.0, f"Expected MI > 1.0, got {mi}"
    assert math.isclose(mi, 1.5, rel_tol=1e-6), f"Expected MI == 1.5, got {mi}"


def test_misalignment_index_s_star_at_last_grid_point():
    """S* at or beyond the last grid point returns None (no usable crossing)."""
    s_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    r_hat = np.ones(5)

    # S* exactly at last grid point
    assert misalignment_index(r_hat, s_grid, 1.0) is None

    # S* beyond the grid (e.g. 0.95 rounded up to nearest grid point = 1.0)
    assert misalignment_index(r_hat, s_grid, 1.0 - 1e-14) is None

    # S* strictly before last grid point should still work
    mi = misalignment_index(r_hat, s_grid, 0.75)
    assert mi is not None
    assert math.isclose(mi, 1.0, abs_tol=1e-5)
