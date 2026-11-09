"""Open-ended generation metrics for reliance-utility evaluation.

Provides:
- Canonical VQA answer string normalization (lowercase, punctuation/article removal).
- VQA soft accuracy: min(matches / 3.0, 1.0) against human direct_answers.
- Token-level sequence Jensen-Shannon divergence (base 2) across answer token positions.
- Answer-change rate tracking against clean reference answer A*.
- Open-ended reliance R(S) and normalized reliance R_hat(S) = R(S) / R(0).
"""

import re
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np


def normalize_answer(s: str) -> str:
    """Canonical VQA answer string normalization.

    1. Lowercase text.
    2. Replace punctuation with spaces.
    3. Remove articles ('a', 'an', 'the').
    4. Normalize whitespace.

    Parameters
    ----------
    s : str
        Raw predicted or reference answer text.

    Returns
    -------
    str
        Normalized canonical answer string.
    """
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = s.lower().strip()
    # Replace punctuation characters with spaces
    s = re.sub(r"[^\w\s]", " ", s)
    # Filter out articles
    tokens = [t for t in s.split() if t not in ("a", "an", "the")]
    return " ".join(tokens)


def vqa_soft_accuracy(prediction: str, direct_answers: Sequence[str]) -> float:
    """Compute VQA 10-annotator soft accuracy metric.

    Acc(pred) = min(count(normalized match) / 3.0, 1.0)

    Parameters
    ----------
    prediction : str
        Model generated answer string.
    direct_answers : Sequence[str]
        List of human ground-truth answers (e.g. 10 annotator direct_answers).

    Returns
    -------
    float
        Soft accuracy in [0.0, 1.0].
    """
    if not direct_answers:
        return 0.0

    norm_pred = normalize_answer(prediction)
    if not norm_pred:
        return 0.0

    matches = 0
    for ans in direct_answers:
        if normalize_answer(ans) == norm_pred:
            matches += 1

    return min(matches / 3.0, 1.0)


def is_answer_changed(prediction: str, reference: str) -> bool:
    """Return True if normalized prediction differs from normalized reference."""
    return normalize_answer(prediction) != normalize_answer(reference)


def token_jsd(
    p: Union[Sequence[float], np.ndarray],
    q: Union[Sequence[float], np.ndarray],
    eps: float = 1e-12,
) -> float:
    """Compute Jensen-Shannon divergence in base 2 between two vocabulary distributions.

    Bounded in [0.0, 1.0].

    Parameters
    ----------
    p : array-like of shape (V,)
        First probability distribution over vocabulary of size V.
    q : array-like of shape (V,)
        Second probability distribution over vocabulary of size V.
    eps : float, default=1e-12
        Smoothing parameter for numerical stability.

    Returns
    -------
    float
        JSD in [0.0, 1.0].
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    if np.allclose(p_arr, q_arr, atol=1e-12):
        return 0.0

    p_arr = np.maximum(p_arr, 0.0)
    q_arr = np.maximum(q_arr, 0.0)

    p_smooth = p_arr + eps
    q_smooth = q_arr + eps

    p_norm = p_smooth / np.sum(p_smooth)
    q_norm = q_smooth / np.sum(q_smooth)

    m = 0.5 * (p_norm + q_norm)
    kl_pm = np.sum(p_norm * np.log2(p_norm / m))
    kl_qm = np.sum(q_norm * np.log2(q_norm / m))

    val = 0.5 * (kl_pm + kl_qm)
    return float(np.clip(val, 0.0, 1.0))


def sequence_jsd(
    p_seq: np.ndarray,
    q_seq: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, List[float]]:
    """Compute token-level Jensen-Shannon divergence across answer positions.

    Parameters
    ----------
    p_seq : np.ndarray of shape (T, V)
        Sequence of T probability distributions over vocabulary V.
    q_seq : np.ndarray of shape (T, V)
        Reference sequence of T probability distributions over vocabulary V.
    eps : float, default=1e-12
        Epsilon smoothing parameter.

    Returns
    -------
    Tuple[float, List[float]]
        (mean_jsd, per_token_jsd_list) where mean_jsd is in [0.0, 1.0].
    """
    p_arr = np.asarray(p_seq, dtype=float)
    q_arr = np.asarray(q_seq, dtype=float)

    if p_arr.ndim != 2 or q_arr.ndim != 2:
        raise ValueError(f"Expected 2D arrays (T, V), got {p_arr.shape} and {q_arr.shape}")

    t_len = min(len(p_arr), len(q_arr))
    if t_len == 0:
        return 0.0, []

    token_jsds: List[float] = []
    for t in range(t_len):
        token_jsds.append(token_jsd(p_arr[t], q_arr[t], eps=eps))

    mean_val = float(np.mean(token_jsds))
    return mean_val, token_jsds


def compute_sequence_reliance(
    p_seq_s: np.ndarray,
    p_seq_null: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute token-level reliance R(S) as mean JSD against null baseline.

    R(S) = (1/T) sum_{t=1}^T D_JS(P_t(S) || P_t(null))
    """
    mean_jsd, _ = sequence_jsd(p_seq_s, p_seq_null, eps=eps)
    return mean_jsd
