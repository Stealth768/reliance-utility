"""Unit tests for open-ended sequence reliance metrics and synthetic sequence validation (CPU tests)."""

import numpy as np
import pytest

from src.metrics.open_ended import (
    compute_sequence_reliance,
    is_answer_changed,
    normalize_answer,
    sequence_jsd,
    token_jsd,
    vqa_soft_accuracy,
)
from src.synthetic.seq import (
    SeqFixedModel,
    SeqGatedModel,
    SeqOracleGatedModel,
    SeqPartialGatedModel,
    generate_synthetic_sequences,
    run_synthetic_seq_validation,
)


# --- 1. Canonical VQA Normalization Tests ---

def test_normalize_answer_casing_and_whitespace():
    """normalize_answer lowercases text and normalizes multiple whitespace."""
    assert normalize_answer("  Hello   World  ") == "hello world"
    assert normalize_answer("RED CAR") == "red car"


def test_normalize_answer_removes_punctuation():
    """normalize_answer replaces punctuation with spaces and collapses whitespace."""
    assert normalize_answer("yes, it is!") == "yes it is"
    assert normalize_answer("stop-sign.") == "stop sign"
    assert normalize_answer("what's that?") == "what s that"


def test_normalize_answer_removes_articles():
    """normalize_answer strips English articles ('a', 'an', 'the')."""
    assert normalize_answer("the dog") == "dog"
    assert normalize_answer("a brown bear") == "brown bear"
    assert normalize_answer("an apple") == "apple"
    assert normalize_answer("the cat sat on a mat") == "cat sat on mat"


def test_normalize_answer_empty():
    """normalize_answer handles empty or non-string inputs safely."""
    assert normalize_answer("") == ""
    assert normalize_answer("    ") == ""
    assert normalize_answer(None) == ""


# --- 2. VQA Soft Accuracy Tests ---

def test_vqa_soft_accuracy_standard():
    """vqa_soft_accuracy computes min(matches / 3.0, 1.0) against direct_answers."""
    annotators = ["cat", "cat", "cat", "dog", "dog", "cat", "kitten", "cat", "cat", "cat"]
    # 7 matches -> min(7/3, 1.0) = 1.0
    assert vqa_soft_accuracy("cat", annotators) == 1.0

    # 2 matches -> 2/3 = 0.6666...
    assert pytest.approx(vqa_soft_accuracy("dog", annotators), rel=1e-4) == 2.0 / 3.0

    # 1 match -> 1/3 = 0.3333...
    assert pytest.approx(vqa_soft_accuracy("kitten", annotators), rel=1e-4) == 1.0 / 3.0

    # 0 matches -> 0.0
    assert vqa_soft_accuracy("elephant", annotators) == 0.0


def test_vqa_soft_accuracy_with_normalization():
    """vqa_soft_accuracy matches after article and punctuation normalization."""
    annotators = ["a red car", "the red car", "red car", "blue car"]
    # "Red Car!" normalizes to "red car", matching 3 annotators -> 3/3 = 1.0
    assert vqa_soft_accuracy("Red Car!", annotators) == 1.0


def test_vqa_soft_accuracy_edge_cases():
    """vqa_soft_accuracy handles empty predictions or annotator lists gracefully."""
    assert vqa_soft_accuracy("", ["cat", "cat"]) == 0.0
    assert vqa_soft_accuracy("cat", []) == 0.0


def test_is_answer_changed():
    """is_answer_changed checks whether normalized strings differ."""
    assert not is_answer_changed("The dog", "a dog")
    assert is_answer_changed("dog", "cat")
    assert not is_answer_changed("red, car.", "Red Car")


# --- 3. Token and Sequence JSD Tests ---

def test_token_jsd_bounds_and_symmetry():
    """token_jsd is symmetric, non-negative, and bounded in [0, 1]."""
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.2, 0.7])

    jsd_pq = token_jsd(p, q)
    jsd_qp = token_jsd(q, p)

    assert pytest.approx(jsd_pq, rel=1e-6) == jsd_qp
    assert 0.0 <= jsd_pq <= 1.0

    # Identical distributions return 0.0
    assert token_jsd(p, p) == 0.0


def test_sequence_jsd_shape_and_mean():
    """sequence_jsd computes token-wise JSD across positions and returns mean."""
    p_seq = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.33, 0.33, 0.34],
    ])
    q_seq = np.array([
        [1.0, 0.0, 0.0],  # identical -> JSD = 0
        [0.0, 0.5, 0.5],
        [0.33, 0.33, 0.34],  # identical -> JSD = 0
    ])

    mean_jsd, token_jsds = sequence_jsd(p_seq, q_seq)
    assert len(token_jsds) == 3
    assert token_jsds[0] == 0.0
    assert token_jsds[2] == 0.0
    assert token_jsds[1] > 0.0
    assert pytest.approx(mean_jsd, rel=1e-6) == float(np.mean(token_jsds))


def test_sequence_jsd_invalid_shape():
    """sequence_jsd raises ValueError if non-2D arrays provided."""
    with pytest.raises(ValueError, match="Expected 2D arrays"):
        sequence_jsd(np.zeros((3,)), np.zeros((3,)))


# --- 4. Synthetic Sequence Benchmark Validation ---

def test_synthetic_sequence_generator():
    """generate_synthetic_sequences generates correct shapes and ranges."""
    data = generate_synthetic_sequences(n_samples=50, seq_len=3, vocab_size=20, seed=123)
    assert data["targets"].shape == (50, 3)
    assert data["clean_logits_A"].shape == (50, 3, 20)
    assert data["logits_B"].shape == (50, 3, 20)
    assert np.all((data["targets"] >= 0) & (data["targets"] < 20))


def test_synthetic_sequence_success_criteria():
    """run_synthetic_seq_validation satisfies Oracle < 0.05 and MI ordering criteria."""
    benchmark = run_synthetic_seq_validation(n_samples=200, cutoff=0.50, seed=42, grid=9)

    # 1. Oracle criterion: R_hat < 0.05 for all S >= cutoff
    assert benchmark["oracle_criterion"] is True, (
        f"Oracle max R_hat beyond cutoff was {benchmark['oracle_max_r_hat_beyond']}, expected < 0.05"
    )

    # 2. MI ordering criterion: MI(Gated) < MI(Partial) < MI(Fixed)
    assert benchmark["mi_ordering_criterion"] is True, (
        f"MI ordering failed: {benchmark['mi_values']}"
    )

    mi = benchmark["mi_values"]
    assert mi["Gated"] < mi["Partial"]
    assert mi["Partial"] < mi["Fixed"]
