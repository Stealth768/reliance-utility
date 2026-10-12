"""Unit tests for A-OKVQA data loader and pilot index sampling (CPU tests)."""

import json
from pathlib import Path
import pytest

from src.vlm.data import (
    get_dataset_metadata,
    load_aokvqa_validation,
    sample_pilot_indices,
)


def test_sample_pilot_indices_deterministic():
    """sample_pilot_indices is deterministic given the seed and returns unique sorted indices."""
    idx1 = sample_pilot_indices(total_items=1000, n=200, seed=42)
    idx2 = sample_pilot_indices(total_items=1000, n=200, seed=42)

    assert len(idx1) == 200
    assert len(set(idx1)) == 200
    assert idx1 == sorted(idx1)
    assert idx1 == idx2


def test_sample_pilot_indices_validation():
    """sample_pilot_indices raises ValueError when n exceeds total_items."""
    with pytest.raises(ValueError, match="exceeds total items"):
        sample_pilot_indices(total_items=50, n=100, seed=0)


def test_get_dataset_metadata():
    """get_dataset_metadata reports correct A-OKVQA dataset id and revision."""
    meta = get_dataset_metadata()
    assert meta["dataset_id"] == "HuggingFaceM4/A-OKVQA"
    assert meta["revision"] == "d1b0efa3a436e9101dfbde3752db7607da696c35"
    assert meta["split"] == "validation"


def test_pilot_ids_file_exists_and_format():
    """results/derived/pilot_ids.json exists, is valid JSON, and contains 200 IDs."""
    path = Path("results/derived/pilot_ids.json")
    assert path.exists(), "pilot_ids.json was not found"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_id"] == "HuggingFaceM4/A-OKVQA"
    assert data["revision"] == "d1b0efa3a436e9101dfbde3752db7607da696c35"
    assert data["n_sample"] == 200
    assert data["seed"] == 0
    assert len(data["question_ids"]) == 200
    assert len(set(data["question_ids"])) == 200


def test_main_ids_disjoint_from_pilot():
    """results/derived/main_ids.json has N=500 unique IDs strictly disjoint from pilot_ids.json."""
    pilot_path = Path("results/derived/pilot_ids.json")
    main_path = Path("results/derived/main_ids.json")

    assert pilot_path.exists(), "pilot_ids.json missing"
    assert main_path.exists(), "main_ids.json missing"

    with open(pilot_path, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)
    with open(main_path, "r", encoding="utf-8") as f:
        main_data = json.load(f)

    pilot_ids = set(pilot_data["question_ids"])
    main_ids = set(main_data["question_ids"])

    assert len(pilot_ids) == 200
    assert len(main_ids) == 500
    assert len(main_data["question_ids"]) == 500  # no duplicates in array

    # Intersection assertion
    intersection = pilot_ids.intersection(main_ids)
    assert len(intersection) == 0, f"Found {len(intersection)} overlapping IDs: {intersection}"


def test_three_id_sets_are_disjoint():
    """pilot_ids, main_ids, and rep_ids are pairwise disjoint and together cover all 1145 validation IDs."""
    pilot_path = Path("results/derived/pilot_ids.json")
    main_path = Path("results/derived/main_ids.json")
    rep_path = Path("results/derived/rep_ids.json")

    for p in (pilot_path, main_path, rep_path):
        assert p.exists(), f"{p} missing"

    with open(pilot_path, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)
    with open(main_path, "r", encoding="utf-8") as f:
        main_data = json.load(f)
    with open(rep_path, "r", encoding="utf-8") as f:
        rep_data = json.load(f)

    pilot_ids = set(pilot_data["question_ids"])
    main_ids = set(main_data["question_ids"])
    rep_ids = set(rep_data["question_ids"])

    # --- Size checks ---
    assert len(pilot_ids) == 200, f"Expected 200 pilot IDs, got {len(pilot_ids)}"
    assert len(main_ids) == 500, f"Expected 500 main IDs, got {len(main_ids)}"
    assert len(rep_ids) == 445, f"Expected 445 rep IDs, got {len(rep_ids)}"

    # No duplicates within each array
    assert len(pilot_data["question_ids"]) == 200
    assert len(main_data["question_ids"]) == 500
    assert len(rep_data["question_ids"]) == 445

    # --- Pairwise disjointness ---
    pilot_main = pilot_ids & main_ids
    assert len(pilot_main) == 0, f"pilot ∩ main not empty: {pilot_main}"

    pilot_rep = pilot_ids & rep_ids
    assert len(pilot_rep) == 0, f"pilot ∩ rep not empty: {pilot_rep}"

    main_rep = main_ids & rep_ids
    assert len(main_rep) == 0, f"main ∩ rep not empty: {main_rep}"

    # --- Union covers full validation set (1145 questions) ---
    total_union = pilot_ids | main_ids | rep_ids
    assert len(total_union) == 1145, (
        f"Union of all three ID sets = {len(total_union)}, expected 1145"
    )

