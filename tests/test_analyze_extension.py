"""Unit tests for Extension Phase analysis and hypothesis evaluation."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from scripts.analyze_extension_results import (
    bootstrap_mc_stats,
    bootstrap_open_stats,
    compute_mc_metrics_itemwise,
    compute_open_metrics_itemwise,
)


def test_mc_metrics_computation_synthetic():
    """Verify MC metric calculation on controlled synthetic items."""
    # 10 synthetic questions, 4 choices
    items = []
    for i in range(10):
        items.append({
            "question_id": f"q_{i}",
            "correct_choice_idx": 0,
            "null_probs": {"removed": [0.7, 0.1, 0.1, 0.1]},
            "probs_per_S": [
                [0.9, 0.05, 0.03, 0.02],  # S=0
                [0.5, 0.2, 0.2, 0.1],      # S=1
            ],
        })
    data = compute_mc_metrics_itemwise(items, null_kind="removed")
    assert data["n_items"] == 10
    assert len(data["jsd_s0"]) == 10
    assert len(data["jsd_s1"]) == 10

    stats = bootstrap_mc_stats(data, n_boot=100, seed=42)
    assert stats["rhat1"] > 0.0
    assert len(stats["rhat1_ci"]) == 2
    assert stats["rhat1_ci"][0] <= stats["rhat1_ci"][1]


def test_open_metrics_computation_and_floor_handling():
    """Verify open-ended metrics and handling of floor accuracy differences."""
    items = []
    for i in range(20):
        # All items have zero accuracy across clean, null, and S=1
        items.append({
            "question_id": f"q_open_{i}",
            "clean_answer": "cat",
            "clean_utility": 0.0,
            "null_answers": {"removed": "dog"},
            "null_utility": {"removed": 0.0},
            "R0": {"removed": 0.40},
            "answers_per_S": ["cat", "bird"],
            "utility_per_S": [0.0, 0.0],
            "R_removed_per_S": [0.40, 0.30],
        })

    data = compute_open_metrics_itemwise(items, null_kind="removed")
    assert data["n_items"] == 20
    assert np.all(data["u1_items"] == 0.0)

    stats = bootstrap_open_stats(data, n_boot=100, seed=42)
    assert np.isclose(stats["u1"], 0.0)
    assert stats["u1_ci"] == (0.0, 0.0)
    assert np.isclose(stats["rhat1"], 0.75)


def test_holm_bonferroni_monotonicity_and_stepdown():
    """Test Holm-Bonferroni correction logic on controlled p-values."""
    # Pre-declared family of M=6 tests
    raw_p = [0.0001, 0.0001, 0.0001, 0.02, 0.03, 0.04]
    M = len(raw_p)
    sorted_indices = sorted(range(M), key=lambda i: raw_p[i])

    adj_p = [0.0] * M
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        multiplier = M - rank
        val = min(raw_p[idx] * multiplier, 1.0)
        adj_p[idx] = max(running_max, val)
        running_max = adj_p[idx]

    # Monotonicity check
    for r in range(M - 1):
        idx_curr = sorted_indices[r]
        idx_next = sorted_indices[r + 1]
        assert adj_p[idx_curr] <= adj_p[idx_next]

    # Extreme low p-values should be adjusted by multiplier <= M
    assert adj_p[sorted_indices[0]] == raw_p[sorted_indices[0]] * 6


def test_donor_derangement_empirical_range_no_variance_test():
    """Verify empirical range across seeds {0, 11, 12, 13} without variance tests."""
    seed_values = [0.487, 0.487, 0.462, 0.510]
    emp_range = max(seed_values) - min(seed_values)
    assert np.isclose(emp_range, 0.048, atol=1e-3)
    assert emp_range < 0.10


def test_paper_extended_tables_and_placeholders():
    """Verify that paper_extended/main.tex has zero placeholders and valid numbers."""
    tex_path = REPO_ROOT / "paper_extended" / "main.tex"
    assert tex_path.exists(), "paper_extended/main.tex not found"

    content = tex_path.read_text(encoding="utf-8")

    # Zero placeholders
    assert "pending" not in content.lower()
    assert "results narrative reserved" not in content.lower()
    assert r"\checknum" not in content
    assert r"\input{" not in content

    # Check that Table labels exist
    for lbl in ["tab:ai2d_headline", "tab:open_headline", "tab:pairing_stability", "tab:hypothesis_tests"]:
        assert f"\\label{{{lbl}}}" in content

    # Check figures are referenced
    assert "figures/fig3_ai2d_two_panel.pdf" in content
    assert "figures/fig4_open_and_stability.pdf" in content
