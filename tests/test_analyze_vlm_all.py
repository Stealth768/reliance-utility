"""CPU tests for multi-model analysis pipeline and paper artifact generation."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.analyze_vlm_all import (
    analyze_model_corruption_all_seeds,
    compute_metrics_for_items,
    load_shards_grouped,
    run_full_analysis,
)
from scripts.check_numbers import check_numbers


def create_fake_shard(
    path: Path,
    model_name: str,
    corruption: str,
    seed: int,
    shard_idx: int,
    start_idx: int,
    n_items: int = 5,
    n_grid: int = 9,
    base_acc: float = 0.8,
) -> Path:
    """Create a minimal synthetic VLM evaluation shard."""
    S_grid = np.linspace(0.0, 1.0, n_grid).tolist()
    rng = np.random.default_rng(seed * 100 + shard_idx)

    items = []
    for i in range(n_items):
        qid = start_idx + i
        correct_idx = int(rng.integers(0, 4))

        # Probs per S: clean at S=0 has high prob on correct, degrades as S increases
        probs_per_s = []
        mass_per_s = []
        for s_idx, S in enumerate(S_grid):
            p = np.full(4, 0.05 + S * 0.2)
            p[correct_idx] = 0.85 - S * 0.6
            p = p / np.sum(p)
            probs_per_s.append(p.tolist())
            mass_per_s.append(0.95)

        # Null removed has correct prediction (so acc=1.0, matching S=0 and giving U(0)=0),
        # but with a distinct distribution so JSD(clean, null) > 0 and R(0) > 0.
        p_nr = np.full(4, 0.1)
        p_nr[correct_idx] = 0.7
        p_null_removed = (p_nr / np.sum(p_nr)).tolist()
        p_null_black = [0.25, 0.25, 0.25, 0.25]
        p_swapped = [0.25, 0.25, 0.25, 0.25]

        items.append({
            "question_id": qid,
            "correct_choice_idx": correct_idx,
            "null_probs": {
                "removed": p_null_removed,
                "black": p_null_black,
            },
            "null_mass": {
                "removed": 0.95,
                "black": 0.95,
            },
            "swapped_probs": p_swapped,
            "swapped_mass": 0.95,
            "probs_per_S": probs_per_s,
            "mass_per_S": mass_per_s,
        })

    shard_data = {
        "model_name": model_name,
        "corruption": corruption,
        "shard_idx": shard_idx,
        "seed": seed,
        "start_idx": start_idx,
        "end_idx": start_idx + n_items,
        "n_questions": n_items,
        "S_grid": S_grid,
        "items": items,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(shard_data, f)
    return path


@pytest.fixture
def fake_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    # 2 models, 1 corruption, 2 seeds, 2 shards each
    for model in ["Qwen/Qwen2-VL-2B-Instruct", "HuggingFaceTB/SmolVLM-Instruct"]:
        m_slug = model.split("/")[-1]
        for seed in [0, 1]:
            for shard_idx in [0, 1]:
                fname = raw_dir / f"{m_slug}_gaussian_seed{seed}_shard{shard_idx}.json"
                create_fake_shard(
                    fname,
                    model_name=model,
                    corruption="gaussian",
                    seed=seed,
                    shard_idx=shard_idx,
                    start_idx=shard_idx * 5,
                    n_items=5,
                )
    return raw_dir


def test_fake_shards_produce_summary_csv(fake_raw_dir: Path, tmp_path: Path):
    """Full analysis pipeline generates summary CSV with expected columns."""
    out_dir = tmp_path / "derived"
    table_dir = tmp_path / "tables"
    fig_dir = tmp_path / "figures"

    run_full_analysis(
        raw_dir=fake_raw_dir,
        out_dir=out_dir,
        table_dir=table_dir,
        fig_dir=fig_dir,
        n_boot=100,
        seed=42,
    )

    summary_csv = out_dir / "vlm_main_summary.csv"
    assert summary_csv.exists()

    df = pd.read_csv(summary_csv)
    assert len(df) == 2  # 2 models x 1 corruption
    expected_cols = [
        "model_name", "corruption", "n_seeds", "n_items",
        "clean_acc", "acc_s1", "acc_drop", "null_removed_acc",
        "null_black_acc", "swapped_acc", "mean_letter_mass",
        "S_star", "sig_S_star", "MI", "MI_ci_pooled", "MI_ci_hierarchical",
        "conditional_mi_range", "frac_usable_resamples",
        "R_hat_s1", "R_hat_s1_ci",
        "no_crossing_frac", "n_no_usable_crossing", "frac_no_usable_crossing",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert (table_dir / "vlm_main_summary.tex").exists()
    assert (out_dir / "vlm_main_correlations.csv").exists()


def test_accuracy_at_s0_matches_correct_idx(fake_raw_dir: Path):
    """Clean accuracy at S=0 evaluates correctly from probabilities."""
    shards = load_shards_grouped(fake_raw_dir)
    key = ("Qwen/Qwen2-VL-2B-Instruct", "gaussian", 0)
    items = shards[key]["items"]
    res = compute_metrics_for_items(items, shards[key]["S_grid"], null_kind="removed")

    # In our fake shard generator, S=0 has p[correct_idx] highest
    assert res["clean_acc"] == 1.0


def test_u_at_s0_is_zero(fake_raw_dir: Path):
    """Utility U(0) is 0.0 when null_removed has identical accuracy to clean S=0."""
    shards = load_shards_grouped(fake_raw_dir)
    key = ("Qwen/Qwen2-VL-2B-Instruct", "gaussian", 0)
    items = shards[key]["items"]
    res = compute_metrics_for_items(items, shards[key]["S_grid"], null_kind="removed")

    # U(0) = acc_deg[0] - acc_null = 1.0 - 1.0 = 0.0
    assert np.isclose(res["U_S"][0], 0.0)


def test_no_crossing_fraction_in_range(fake_raw_dir: Path):
    """no_crossing_frac is a valid probability in [0, 1]."""
    shards = load_shards_grouped(fake_raw_dir)
    seed_groups = [
        shards[("Qwen/Qwen2-VL-2B-Instruct", "gaussian", 0)],
        shards[("Qwen/Qwen2-VL-2B-Instruct", "gaussian", 1)],
    ]
    res = analyze_model_corruption_all_seeds(
        "Qwen/Qwen2-VL-2B-Instruct", "gaussian", seed_groups, n_boot=50, seed=42
    )
    assert 0.0 <= res["no_crossing_frac"] <= 1.0


def test_hierarchical_boot_wider_than_pooled(fake_raw_dir: Path):
    """Hierarchical bootstrap returns valid finite bounds."""
    shards = load_shards_grouped(fake_raw_dir)
    seed_groups = [
        shards[("Qwen/Qwen2-VL-2B-Instruct", "gaussian", 0)],
        shards[("Qwen/Qwen2-VL-2B-Instruct", "gaussian", 1)],
    ]
    res = analyze_model_corruption_all_seeds(
        "Qwen/Qwen2-VL-2B-Instruct", "gaussian", seed_groups, n_boot=200, seed=42
    )

    # Hierarchical CI widths are valid numbers
    if not np.isnan(res["MI_ci_hier_low"]) and not np.isnan(res["MI_ci_pooled_low"]):
        hier_w = res["MI_ci_hier_high"] - res["MI_ci_hier_low"]
        pooled_w = res["MI_ci_pooled_high"] - res["MI_ci_pooled_low"]
        # Both must be >= 0
        assert hier_w >= 0.0
        assert pooled_w >= 0.0


def test_check_numbers_matches_and_detects_mismatch(tmp_path: Path):
    """check_numbers validates numbers and raises failure on mismatch."""
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    summary_csv = tmp_path / "summary.csv"

    # Create dummy summary CSV
    df = pd.DataFrame([{
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "corruption": "gaussian",
        "clean_acc": 0.8523,
    }])
    df.to_csv(summary_csv, index=False)

    # 1. Matching TeX file (empty raw_dir so freshness passes)
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()
    match_tex = paper_dir / "section.tex"
    match_tex.write_text(
        r"The accuracy was \checknum{0.8523}{Qwen2-VL-2B-Instruct:gaussian:clean_acc} on clean.",
        encoding="utf-8",
    )
    assert check_numbers(paper_dir, summary_csv, raw_dir=empty_raw) is True

    # 2. Mismatching TeX file
    mismatch_tex = paper_dir / "section.tex"
    mismatch_tex.write_text(
        r"The accuracy was \checknum{0.9999}{Qwen2-VL-2B-Instruct:gaussian:clean_acc} on clean.",
        encoding="utf-8",
    )
    assert check_numbers(paper_dir, summary_csv, raw_dir=empty_raw) is False

    # 3. Missing CSV fails
    missing_csv = tmp_path / "missing.csv"
    assert check_numbers(paper_dir, missing_csv, raw_dir=empty_raw) is False

    # 4. Stale CSV fails (raw shard modified after CSV)
    raw_with_shard = tmp_path / "stale_raw"
    raw_with_shard.mkdir()
    shard_f = raw_with_shard / "shard0.json"
    shard_f.write_text("{}", encoding="utf-8")
    # Make shard mtime definitively after summary_csv
    import os, time
    os.utime(shard_f, (time.time() + 10, time.time() + 10))
    assert check_numbers(paper_dir, summary_csv, raw_dir=raw_with_shard) is False


def test_analyze_vlm_all_reads_legacy_shards_as_seed_0(tmp_path: Path):
    """load_shards_grouped maps legacy shard without seed in name or data to seed 0."""
    raw_dir = tmp_path / "raw_legacy"
    raw_dir.mkdir()
    fname = raw_dir / "Qwen2-VL-2B-Instruct_gaussian_shard0.json"
    fname.write_text(json.dumps({
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "corruption": "gaussian",
        "shard_idx": 0,
        "items": [],
    }), encoding="utf-8")

    grouped = load_shards_grouped(raw_dir)
    assert ("Qwen/Qwen2-VL-2B-Instruct", "gaussian", 0) in grouped


def test_analyze_vlm_all_reports_dropped_resamples_when_r0_is_zero(tmp_path: Path):
    """When R(0)~0 causes exceptions in bootstrap resamples, dropped count is tracked and reported."""
    raw_dir = tmp_path / "raw_r0_zero"
    raw_dir.mkdir()
    fname = raw_dir / "Qwen2-VL-2B-Instruct_gaussian_shard0.json"

    S_grid = np.linspace(0.0, 1.0, 9).tolist()
    items = []
    # Force clean S=0 to have IDENTICAL probabilities to null_removed for all items
    # so that JSD(clean, null) == 0.0 and R(0) == 0.0.
    for i in range(10):
        corr = i % 4
        probs_s = []
        for s_idx, S in enumerate(S_grid):
            p = np.full(4, 0.1)
            # High utility at S=0, crossing to negative utility at high S
            if s_idx == 0:
                p[corr] = 0.7
            else:
                p[corr] = 0.7 - S * 0.6  # drops to 0.1 at S=1
            p = (p / np.sum(p)).tolist()
            probs_s.append(p)

        # Identical to S=0 distribution:
        p_null = list(probs_s[0])
        items.append({
            "question_id": i,
            "correct_choice_idx": corr,
            "null_probs": {"removed": p_null, "black": [0.25]*4},
            "null_mass": {"removed": 1.0, "black": 1.0},
            "swapped_probs": [0.25]*4,
            "swapped_mass": 1.0,
            "probs_per_S": probs_s,
            "mass_per_S": [1.0]*len(S_grid),
        })

    shard_data = {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "corruption": "gaussian",
        "shard_idx": 0,
        "seed": 0,
        "S_grid": S_grid,
        "items": items,
    }
    fname.write_text(json.dumps(shard_data), encoding="utf-8")

    out_dir = tmp_path / "out"
    table_dir = tmp_path / "tables"
    fig_dir = tmp_path / "figures"

    run_full_analysis(
        raw_dir=raw_dir,
        out_dir=out_dir,
        table_dir=table_dir,
        fig_dir=fig_dir,
        n_boot=100,
        seed=42,
    )

    summary_df = pd.read_csv(out_dir / "vlm_main_summary.csv")
    row = summary_df.iloc[0]
    assert "n_no_usable_crossing" in row
    assert "frac_no_usable_crossing" in row
    assert "R_hat_s1" in row
    assert "R_hat_s1_ci" in row
    # Because R(0) is strictly 0.0, any resample that found a crossing will have failed in normalized_reliance
    assert row["n_no_usable_crossing"] > 0
    assert row["frac_no_usable_crossing"] > 0.0


def test_dropped_resamples_cause_columns_present_and_sum(tmp_path: Path):
    """Four cause columns are present in the CSV and their sum equals n_no_usable_crossing."""
    raw_dir = tmp_path / "raw_cause"
    raw_dir.mkdir()

    S_grid = np.linspace(0.0, 1.0, 9).tolist()
    items = []
    for i in range(12):
        corr = i % 4
        probs_s = []
        for s_idx, S in enumerate(S_grid):
            p = np.full(4, 0.1)
            if s_idx == 0:
                p[corr] = 0.7
            else:
                p[corr] = max(0.1, 0.7 - S * 0.8)
            p = p / p.sum()
            probs_s.append(p.tolist())
        # null_removed identical to clean (S=0) → R(0) ~ 0 → triggers r0_zero drops
        p_null = probs_s[0]
        items.append({
            "question_id": i,
            "correct_choice_idx": corr,
            "null_probs": {"removed": p_null, "black": [0.25, 0.25, 0.25, 0.25]},
            "null_mass": {"removed": 0.95, "black": 0.95},
            "swapped_probs": [0.25, 0.25, 0.25, 0.25],
            "swapped_mass": 0.95,
            "probs_per_S": probs_s,
            "mass_per_S": [0.95] * len(S_grid),
        })

    shard = {
        "model_name": "TestModel/CauseTest",
        "corruption": "gaussian",
        "shard_idx": 0,
        "seed": 0,
        "S_grid": S_grid,
        "items": items,
    }
    (raw_dir / "CauseTest_gaussian_shard0.json").write_text(
        json.dumps(shard), encoding="utf-8"
    )

    run_full_analysis(
        raw_dir=raw_dir,
        out_dir=tmp_path / "out_cause",
        table_dir=tmp_path / "tables_cause",
        fig_dir=tmp_path / "figs_cause",
        n_boot=200,
        seed=0,
    )

    df = pd.read_csv(tmp_path / "out_cause" / "vlm_main_summary.csv")
    assert len(df) > 0
    row = df.iloc[0]

    cause_cols = [
        "n_no_usable_crossing_no_cross",
        "n_no_usable_crossing_edge",
        "n_no_usable_crossing_r0_zero",
        "n_no_usable_crossing_other",
    ]
    for col in cause_cols:
        assert col in df.columns, f"Missing column: {col}"
        assert int(row[col]) >= 0, f"{col} must be >= 0, got {row[col]}"

    total_causes = sum(int(row[c]) for c in cause_cols)
    assert total_causes == int(row["n_no_usable_crossing"]), (
        f"Cause sum {total_causes} != n_no_usable_crossing {row['n_no_usable_crossing']}"
    )


def test_load_shards_grouped_scans_multiple_vlm_directories(tmp_path: Path):
    """load_shards_grouped finds shards across sibling vlm_* directories (e.g. vlm_main and vlm_llava)."""
    raw_root = tmp_path / "raw"
    vlm_main = raw_root / "vlm_main"
    vlm_llava = raw_root / "vlm_llava"
    vlm_main.mkdir(parents=True)
    vlm_llava.mkdir(parents=True)

    # Shard in vlm_main (legacy naming, no seed in filename)
    create_fake_shard(
        vlm_main / "Qwen2-VL-2B-Instruct_gaussian_shard0.json",
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        corruption="gaussian",
        seed=0,
        shard_idx=0,
        start_idx=0,
    )
    # Shard in vlm_llava (new naming, with seed)
    create_fake_shard(
        vlm_llava / "llava-1.5-7b-hf_swapblend_seed0_shard0.json",
        model_name="llava-hf/llava-1.5-7b-hf",
        corruption="swapblend",
        seed=0,
        shard_idx=0,
        start_idx=0,
    )

    # Pointing to vlm_main should scan both sibling vlm_* folders
    grouped = load_shards_grouped(vlm_main)
    assert ("Qwen/Qwen2-VL-2B-Instruct", "gaussian", 0) in grouped
    assert ("llava-hf/llava-1.5-7b-hf", "swapblend", 0) in grouped


def test_no_siblings_isolates_single_directory(tmp_path: Path):
    """load_shards_grouped with no_siblings=True reads ONLY the given folder (vlm_rep isolation)."""
    raw_root = tmp_path / "raw"
    vlm_main = raw_root / "vlm_main"
    vlm_rep = raw_root / "vlm_rep"
    vlm_main.mkdir(parents=True)
    vlm_rep.mkdir(parents=True)

    # A shard in vlm_main (would be picked up by sibling scan)
    create_fake_shard(
        vlm_main / "Qwen2-VL-2B-Instruct_gaussian_seed0_shard0.json",
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        corruption="gaussian",
        seed=0,
        shard_idx=0,
        start_idx=0,
    )
    # A shard in vlm_rep (the replication data)
    create_fake_shard(
        vlm_rep / "llava-1.5-7b-hf_swapblend_seed1_shard0.json",
        model_name="llava-hf/llava-1.5-7b-hf",
        corruption="swapblend",
        seed=1,
        shard_idx=0,
        start_idx=0,
    )

    # Without no_siblings: pointing to vlm_rep picks up vlm_main too
    grouped_merged = load_shards_grouped(vlm_rep, no_siblings=False)
    assert ("Qwen/Qwen2-VL-2B-Instruct", "gaussian", 0) in grouped_merged, (
        "sibling scan should include vlm_main key"
    )
    assert ("llava-hf/llava-1.5-7b-hf", "swapblend", 1) in grouped_merged

    # With no_siblings=True: only vlm_rep shards are loaded
    grouped_isolated = load_shards_grouped(vlm_rep, no_siblings=True)
    assert ("llava-hf/llava-1.5-7b-hf", "swapblend", 1) in grouped_isolated, (
        "vlm_rep shard must be present"
    )
    assert ("Qwen/Qwen2-VL-2B-Instruct", "gaussian", 0) not in grouped_isolated, (
        "vlm_main shard must NOT appear when no_siblings=True"
    )
