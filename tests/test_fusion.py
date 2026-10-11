"""CPU unit tests for modality-dropout fusion case study."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.fusion.data import (
    apply_modality_a_degradation,
    generate_synthetic_twoview_data,
    split_data,
    SyntheticTwoViewDataset,
)
from src.fusion.model import TORCH_AVAILABLE, LateFusionModel
from scripts.analyze_fusion import (
    analyze_dataset_p_condition,
    run_full_fusion_analysis,
)


def test_synthetic_dataset_split_reproducible():
    """generate_synthetic_twoview_data produces deterministic splits with fixed seed."""
    d1 = generate_synthetic_twoview_data(n_samples=500, seed=42)
    d2 = generate_synthetic_twoview_data(n_samples=500, seed=42)
    d3 = generate_synthetic_twoview_data(n_samples=500, seed=99)

    assert np.array_equal(d1["x_a"], d2["x_a"])
    assert np.array_equal(d1["x_b"], d2["x_b"])
    assert np.array_equal(d1["labels"], d2["labels"])
    assert not np.array_equal(d1["x_a"], d3["x_a"])

    tr1, val1, te1 = split_data(d1, n_train=300, n_val=100, n_test=100)
    tr2, val2, te2 = split_data(d2, n_train=300, n_val=100, n_test=100)
    assert np.array_equal(tr1["x_a"], tr2["x_a"])
    assert np.array_equal(val1["labels"], val2["labels"])
    assert np.array_equal(te1["x_b"], te2["x_b"])
    assert len(tr1["x_a"]) == 300
    assert len(val1["x_a"]) == 100
    assert len(te1["x_a"]) == 100


def test_dropout_mask_independent_per_sample():
    """Modality dropout is applied independently per sample matching probability p."""
    p = 0.5
    raw = generate_synthetic_twoview_data(n_samples=1000, seed=123)
    ds = SyntheticTwoViewDataset(raw, dropout_p=p, seed=456)

    dropped_flags = [ds[i][3] for i in range(1000)]
    dropped_count = sum(bool(f) for f in dropped_flags)

    # Chi-squared test against binomial null with p=0.5
    # Expected: 500, 500
    res = stats.binomtest(dropped_count, n=1000, p=0.5)
    # p-value must be > 0.01 (not significantly different from 0.5)
    assert res.pvalue > 0.01, f"Dropout count {dropped_count}/1000 significantly diverged from p=0.5"


def test_modality_a_degradation_preserves_shape():
    """apply_modality_a_degradation interpolates smoothly without shape distortion."""
    x_a = np.ones((50, 64), dtype=np.float32)
    deg_0 = apply_modality_a_degradation(x_a, S=0.0, seed=1)
    deg_half = apply_modality_a_degradation(x_a, S=0.5, seed=1)
    deg_1 = apply_modality_a_degradation(x_a, S=1.0, seed=1)

    assert deg_0.shape == (50, 64)
    assert deg_half.shape == (50, 64)
    assert deg_1.shape == (50, 64)
    assert np.array_equal(deg_0, x_a)
    assert not np.array_equal(deg_half, x_a)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available for model parameter test")
def test_null_token_replaces_modality_a():
    """When a_removed=True, LateFusionModel replaces input with learned null token."""
    import torch

    torch.manual_seed(42)
    model = LateFusionModel(dim_a=16, dim_b=16, hidden_dim=16, num_classes=3)
    xa1 = torch.randn(2, 16)
    xa2 = torch.randn(2, 16)
    xb = torch.randn(2, 16)

    # With a_removed=True, different xa inputs should yield IDENTICAL outputs
    out1, _ = model(xa1, xb, a_removed=True)
    out2, _ = model(xa2, xb, a_removed=True)
    assert torch.allclose(out1, out2, atol=1e-6)

    # With a_removed=False, different xa inputs yield DIFFERENT outputs
    out_diff1, _ = model(xa1, xb, a_removed=False)
    out_diff2, _ = model(xa2, xb, a_removed=False)
    assert not torch.allclose(out_diff1, out_diff2, atol=1e-4)


def test_analyze_fusion_on_fake_outputs(tmp_path: Path):
    """analyze_fusion processes mock prediction files and outputs summary table."""
    raw_dir = tmp_path / "fusion_raw"
    out_dir = tmp_path / "fusion_derived"
    table_dir = tmp_path / "fusion_tables"
    fig_dir = tmp_path / "fusion_figs"

    S_grid = np.linspace(0.0, 1.0, 9).tolist()

    # Create fake predictions for 2 p-values (0.0 and 0.25) across 2 seeds
    for p in [0.0, 0.25]:
        for seed in [0, 1]:
            exp_dir = raw_dir / f"synthetic_p{p}_seed{seed}"
            exp_dir.mkdir(parents=True)
            pred_file = exp_dir / "predictions.json"

            items = []
            for i in range(20):
                corr = i % 4
                probs_S = []
                for S in S_grid:
                    pr = np.full(4, 0.1)
                    pr[corr] = 0.7 - S * 0.4
                    pr = (pr / np.sum(pr)).tolist()
                    probs_S.append(pr)

                p_null = np.full(4, 0.2)
                p_null[corr] = 0.4
                p_null = (p_null / np.sum(p_null)).tolist()

                items.append({
                    "item_id": i,
                    "correct_choice_idx": corr,
                    "null_probs": {"removed": p_null},
                    "probs_per_S": probs_S,
                })

            pred_data = {
                "dataset": "synthetic",
                "dropout_p": p,
                "seed": seed,
                "n_test": 20,
                "S_grid": S_grid,
                "items": items,
            }
            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(pred_data, f)

    run_full_fusion_analysis(
        raw_dir=raw_dir,
        out_dir=out_dir,
        table_dir=table_dir,
        fig_dir=fig_dir,
        n_boot=50,
        seed=42,
    )

    summary_csv = out_dir / "fusion_summary.csv"
    assert summary_csv.exists()
    df = pd.read_csv(summary_csv)
    assert len(df) == 2
    assert "dropout_p" in df.columns
    assert "clean_acc" in df.columns
    assert "MI" in df.columns
    assert (table_dir / "fusion_summary.tex").exists()
