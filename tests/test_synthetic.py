"""Unit tests for synthetic benchmark data generator and models."""

import numpy as np
import pytest

from src.synthetic.data import degrade_A, make_data, null_A
from src.synthetic.models import FixedModel, GatedModel, PartialGatedModel


def test_data_generator_reproducibility_and_shapes():
    """Data generator is reproducible by seed and shapes are correct."""
    d1 = make_data(n=200, k=4, d=8, seed=42)
    d2 = make_data(n=200, k=4, d=8, seed=42)

    assert np.array_equal(d1.X_A, d2.X_A)
    assert np.array_equal(d1.X_B, d2.X_B)
    assert np.array_equal(d1.y, d2.y)
    assert np.array_equal(d1.mu_A, d2.mu_A)
    assert np.array_equal(d1.mu_B, d2.mu_B)

    assert d1.X_A.shape == (200, 8)
    assert d1.X_B.shape == (200, 8)
    assert d1.y.shape == (200,)
    assert d1.mu_A.shape == (4, 8)
    assert d1.mu_B.shape == (4, 8)


def test_null_A_variants():
    """null_A generates correct zeros, dataset mean, and removed variants."""
    d = make_data(n=100, k=4, d=8, seed=42)

    # 'removed' returns None
    assert null_A("removed", d.X_A) is None
    assert null_A("removed") is None

    zeros = null_A("zeros", d.X_A)
    assert zeros.shape == (100, 8)
    assert np.all(zeros == 0.0)

    mean_A = null_A("mean", d.X_A)
    assert mean_A.shape == (100, 8)
    expected_mean = np.mean(d.X_A, axis=0)
    assert np.allclose(mean_A[0], expected_mean)


def test_clean_accuracy_and_b_only_range():
    """Clean accuracy of FixedModel > 0.9 and B-only accuracy in 0.5-0.75 range."""
    train_data = make_data(n=2000, k=4, d=8, seed=10)
    test_data = make_data(
        n=2000, k=4, d=8, seed=11, mu_A=train_data.mu_A, mu_B=train_data.mu_B
    )

    model = FixedModel()
    model.fit(train_data.X_A, train_data.X_B, train_data.y)

    clean_acc = float(np.mean(model.predict(test_data.X_A, test_data.X_B) == test_data.y))
    b_only_acc = float(np.mean(model.predict(None, test_data.X_B) == test_data.y))

    assert clean_acc > 0.9, f"Expected clean accuracy > 0.9, got {clean_acc:.3f}"
    assert 0.50 <= b_only_acc <= 0.75, f"Expected B-only accuracy in [0.5, 0.75], got {b_only_acc:.3f}"


def test_fixed_model_utility_negative_at_s1():
    """At S=1 (pure noise), FixedModel utility U < 0."""
    train_data = make_data(n=2000, k=4, d=8, seed=20)
    test_data = make_data(
        n=2000, k=4, d=8, seed=21, mu_A=train_data.mu_A, mu_B=train_data.mu_B
    )

    model = FixedModel()
    model.fit(train_data.X_A, train_data.X_B, train_data.y)

    X_A_noise = degrade_A(test_data.X_A, S=1.0, seed=99)
    acc_noise = float(np.mean(model.predict(X_A_noise, test_data.X_B) == test_data.y))
    acc_null = float(np.mean(model.predict(None, test_data.X_B) == test_data.y))

    u_noise = acc_noise - acc_null
    assert u_noise < 0.0, f"Expected U(1) < 0, got {u_noise:.3f} (noise={acc_noise:.3f}, null={acc_null:.3f})"


def test_gate_decreases_on_average_as_s_increases():
    """Gate decreases on average as S increases."""
    train_data = make_data(n=2000, k=4, d=8, seed=30)
    val_data = make_data(n=1000, k=4, d=8, seed=31)
    test_data = make_data(n=2000, k=4, d=8, seed=32)

    model = GatedModel()
    model.fit(train_data.X_A, train_data.X_B, train_data.y)
    model.calibrate_gate(val_data.X_A, seed=33)

    S_grid = np.linspace(0.0, 1.0, 11)
    mean_gates = []
    for i, S in enumerate(S_grid):
        X_deg = degrade_A(test_data.X_A, S=S, seed=100 + i)
        g = model.compute_gate(X_deg)
        mean_gates.append(float(np.mean(g)))

    # Linear regression slope must be negative
    slope, _ = np.polyfit(S_grid, mean_gates, 1)
    assert slope < 0.0, f"Expected negative slope for gate as S increases, got slope={slope:.3f}"
    assert mean_gates[0] > mean_gates[-1], f"Expected gate(0) > gate(1), got {mean_gates[0]:.3f} vs {mean_gates[-1]:.3f}"


def test_pooled_bootstrap_point_within_ci():
    """Assert point estimate lies within its own pooled bootstrap confidence interval (lo <= point <= hi)."""
    from src.synthetic.run_validation import generate_pooled_summary
    from pathlib import Path

    raw_dir = Path("results/raw/synthetic")
    if not raw_dir.exists() or len(list(raw_dir.glob("*.json"))) == 0:
        pytest.skip("No synthetic raw files to test aggregation on.")

    # Aggregate using pooled bootstrap into a temp or test file
    test_summary_path = Path("results/derived/.test_pooled_summary.csv")
    df = generate_pooled_summary(raw_dir, test_summary_path, n_boot=200, seed=123)

    try:
        for idx, row in df.iterrows():
            if not np.isnan(row["S_star"]) and not np.isnan(row["S_star_lo"]):
                assert row["S_star_lo"] <= row["S_star"] + 1e-9, (
                    f"Row {idx} ({row['model']}/{row['null_kind']}): S_star {row['S_star']} < lo {row['S_star_lo']}"
                )
                assert row["S_star"] <= row["S_star_hi"] + 1e-9, (
                    f"Row {idx} ({row['model']}/{row['null_kind']}): S_star {row['S_star']} > hi {row['S_star_hi']}"
                )

            if not np.isnan(row["MI"]) and not np.isnan(row["MI_lo"]):
                assert row["MI_lo"] <= row["MI"] + 1e-9, (
                    f"Row {idx} ({row['model']}/{row['null_kind']}): MI {row['MI']} < lo {row['MI_lo']}"
                )
                assert row["MI"] <= row["MI_hi"] + 1e-9, (
                    f"Row {idx} ({row['model']}/{row['null_kind']}): MI {row['MI']} > hi {row['MI_hi']}"
                )
    finally:
        if test_summary_path.exists():
            test_summary_path.unlink()

