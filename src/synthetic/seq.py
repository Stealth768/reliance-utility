"""Synthetic sequence fusion benchmark for open-ended reliance validation.

Simulates sequence generation of length L=3 over vocabulary size V=20.
Implements four sequence fusion architectures:
- SeqFixedModel: Quality-blind sequence model with fixed modality weights.
- SeqPartialGatedModel: Model that attenuates visual contribution but retains partial weight.
- SeqGatedModel: Quality-aware sequence model with adaptive sigmoid gate.
- SeqOracleGatedModel: Oracle control explicitly receiving S and shutting off modality A for S >= cutoff.

Success criteria:
1. Oracle achieves normalized reliance R_hat(S) < 0.05 for all S >= cutoff (S* = 0.5).
2. Misalignment Index (MI) ordering: MI(Gated) < MI(Partial) < MI(Fixed).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.special import softmax

from src.metrics.open_ended import sequence_jsd


class SeqFixedModel:
    """Quality-blind sequence fusion model with fixed logit weights."""

    def __init__(self, w_A: float = 1.0, w_B: float = 1.0):
        self.w_A = float(w_A)
        self.w_B = float(w_B)

    def predict_probs(
        self,
        logits_A: Optional[np.ndarray],
        logits_B: np.ndarray,
        S: Optional[float] = None,
    ) -> np.ndarray:
        """Compute softmax probability sequence over vocabulary.

        Parameters
        ----------
        logits_A : Optional[np.ndarray]
            Visual logit sequence of shape (T, V), or None if removed.
        logits_B : np.ndarray
            Language/prior logit sequence of shape (T, V).
        S : Optional[float]
            Severity parameter (ignored by fixed model).

        Returns
        -------
        np.ndarray
            Probability sequence of shape (T, V).
        """
        if logits_A is None:
            fused = self.w_B * logits_B
        else:
            fused = self.w_A * logits_A + self.w_B * logits_B
        return softmax(fused, axis=-1)


class SeqPartialGatedModel:
    """Sequence model that attenuates modality A but retains a non-zero floor weight."""

    def __init__(self, w_A: float = 1.0, w_B: float = 1.0, floor: float = 0.35):
        self.w_A = float(w_A)
        self.w_B = float(w_B)
        self.floor = float(floor)

    def predict_probs(
        self,
        logits_A: Optional[np.ndarray],
        logits_B: np.ndarray,
        S: Optional[float] = None,
    ) -> np.ndarray:
        if logits_A is None:
            fused = self.w_B * logits_B
        else:
            s_val = S if S is not None else 0.0
            gate = max(self.floor, 1.0 - 0.65 * s_val)
            fused = (gate * self.w_A) * logits_A + self.w_B * logits_B
        return softmax(fused, axis=-1)


class SeqGatedModel:
    """Quality-aware sequence model with calibrated sigmoid gating."""

    def __init__(self, w_A: float = 1.0, w_B: float = 1.0, midpoint: float = 0.45, steepness: float = 12.0):
        self.w_A = float(w_A)
        self.w_B = float(w_B)
        self.midpoint = float(midpoint)
        self.steepness = float(steepness)

    def predict_probs(
        self,
        logits_A: Optional[np.ndarray],
        logits_B: np.ndarray,
        S: Optional[float] = None,
    ) -> np.ndarray:
        if logits_A is None:
            fused = self.w_B * logits_B
        else:
            s_val = S if S is not None else 0.0
            gate = 1.0 / (1.0 + np.exp(self.steepness * (s_val - self.midpoint)))
            fused = (gate * self.w_A) * logits_A + self.w_B * logits_B
        return softmax(fused, axis=-1)


class SeqOracleGatedModel:
    """Idealized oracle control model explicitly receiving S and shutting off modality A for S >= cutoff."""

    def __init__(self, w_A: float = 1.0, w_B: float = 1.0, cutoff: float = 0.50):
        self.w_A = float(w_A)
        self.w_B = float(w_B)
        self.cutoff = float(cutoff)

    def predict_probs(
        self,
        logits_A: Optional[np.ndarray],
        logits_B: np.ndarray,
        S: Optional[float] = None,
    ) -> np.ndarray:
        if logits_A is None:
            fused = self.w_B * logits_B
        else:
            s_val = S if S is not None else 0.0
            gate = 1.0 if s_val < self.cutoff else 0.0
            fused = (gate * self.w_A) * logits_A + self.w_B * logits_B
        return softmax(fused, axis=-1)


def generate_synthetic_sequences(
    n_samples: int = 300,
    seq_len: int = 3,
    vocab_size: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate synthetic multimodal sequences for benchmark validation.

    Parameters
    ----------
    n_samples : int
        Number of sequence evaluation items.
    seq_len : int
        Number of tokens in target sequence (default: 3).
    vocab_size : int
        Vocabulary size (default: 20).
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'targets', 'clean_logits_A', 'logits_B', and noise generators.
    """
    rng = np.random.default_rng(seed)

    # Random target sequences of shape (N, T)
    targets = rng.integers(0, vocab_size, size=(n_samples, seq_len))

    # Clean modality A: strong signal on target tokens (+4.0 logit boost)
    clean_logits_A = rng.normal(0.0, 0.5, size=(n_samples, seq_len, vocab_size))
    for i in range(n_samples):
        for t in range(seq_len):
            clean_logits_A[i, t, targets[i, t]] += 4.5

    # Modality B: moderate signal on target tokens (+1.5 logit boost)
    logits_B = rng.normal(0.0, 0.5, size=(n_samples, seq_len, vocab_size))
    for i in range(n_samples):
        for t in range(seq_len):
            logits_B[i, t, targets[i, t]] += 1.5

    # Fixed deceptive noise tensors for corruption
    noise_A = rng.normal(0.0, 1.0, size=(n_samples, seq_len, vocab_size))
    # Deceptive bias on incorrect tokens
    for i in range(n_samples):
        for t in range(seq_len):
            wrong_token = (targets[i, t] + 1) % vocab_size
            noise_A[i, t, wrong_token] += 3.5

    return {
        "targets": targets,
        "clean_logits_A": clean_logits_A,
        "logits_B": logits_B,
        "noise_A": noise_A,
        "n_samples": n_samples,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
    }


def degrade_logits_A(
    clean_logits_A: np.ndarray,
    noise_A: np.ndarray,
    S: float,
) -> np.ndarray:
    """Apply degradation severity S to modality A logits."""
    return (1.0 - S) * clean_logits_A + S * noise_A


def run_synthetic_seq_validation(
    n_samples: int = 300,
    cutoff: float = 0.50,
    seed: int = 42,
    grid: int = 9,
) -> Dict[str, Any]:
    """Execute synthetic sequence benchmark across the four architectures.

    Returns metrics, curves, and validation of success criteria.
    """
    data = generate_synthetic_sequences(n_samples=n_samples, seq_len=3, vocab_size=20, seed=seed)
    targets = data["targets"]
    clean_A = data["clean_logits_A"]
    logits_B = data["logits_B"]
    noise_A = data["noise_A"]

    models = {
        "Fixed": SeqFixedModel(),
        "Partial": SeqPartialGatedModel(floor=0.35),
        "Gated": SeqGatedModel(midpoint=0.45, steepness=12.0),
        "Oracle": SeqOracleGatedModel(cutoff=cutoff),
    }

    S_grid = np.linspace(0.0, 1.0, grid)
    results: Dict[str, Any] = {}

    for name, model in models.items():
        # 1. Null condition (removed A)
        null_probs = np.zeros((n_samples, 3, 20))
        null_correct = 0
        for i in range(n_samples):
            p_null = model.predict_probs(None, logits_B[i], S=None)
            null_probs[i] = p_null
            pred_tokens = np.argmax(p_null, axis=-1)
            null_correct += np.mean(pred_tokens == targets[i])
        u_null = null_correct / n_samples

        # 2. Clean condition (S=0) for reference reliance R(0)
        clean_probs = np.zeros((n_samples, 3, 20))
        for i in range(n_samples):
            clean_probs[i] = model.predict_probs(clean_A[i], logits_B[i], S=0.0)

        # Baseline reliance R(0)
        r0_list = []
        for i in range(n_samples):
            m_jsd, _ = sequence_jsd(clean_probs[i], null_probs[i])
            r0_list.append(m_jsd)
        R0 = float(np.mean(r0_list))

        # 3. Across severity grid S
        u_curve = []
        r_hat_curve = []
        r_raw_curve = []

        for S in S_grid:
            degraded_A = degrade_logits_A(clean_A, noise_A, S=float(S))
            u_items = []
            r_items = []

            for i in range(n_samples):
                p_s = model.predict_probs(degraded_A[i], logits_B[i], S=float(S))
                pred_tokens = np.argmax(p_s, axis=-1)
                u_items.append(np.mean(pred_tokens == targets[i]))
                m_jsd, _ = sequence_jsd(p_s, null_probs[i])
                r_items.append(m_jsd)

            mean_u = float(np.mean(u_items))
            mean_r = float(np.mean(r_items))
            u_curve.append(mean_u)
            r_raw_curve.append(mean_r)
            r_hat_curve.append(mean_r / max(R0, 1e-8))

        # Compute Misalignment Index (MI): integral of R_hat(S) over S in [cutoff, 1.0]
        mask_beyond = S_grid >= cutoff
        S_beyond = S_grid[mask_beyond]
        r_beyond = np.array(r_hat_curve)[mask_beyond]
        # Trapezoidal rule over [cutoff, 1.0]
        if len(S_beyond) > 1:
            mi_val = float(np.trapezoid(r_beyond, S_beyond))
        else:
            mi_val = float(r_beyond[0] * (1.0 - cutoff))

        results[name] = {
            "u_null": u_null,
            "R0": R0,
            "u_curve": u_curve,
            "r_hat_curve": r_hat_curve,
            "r_raw_curve": r_raw_curve,
            "MI": mi_val,
        }

    # Evaluate Success Criteria
    oracle_r_hat_beyond = np.array(results["Oracle"]["r_hat_curve"])[S_grid >= cutoff]
    oracle_criterion = bool(np.all(oracle_r_hat_beyond < 0.05))

    mi_gated = results["Gated"]["MI"]
    mi_partial = results["Partial"]["MI"]
    mi_fixed = results["Fixed"]["MI"]
    mi_ordering_criterion = bool(mi_gated < mi_partial < mi_fixed)

    return {
        "S_grid": S_grid.tolist(),
        "cutoff": cutoff,
        "results": results,
        "oracle_criterion": oracle_criterion,
        "oracle_max_r_hat_beyond": float(np.max(oracle_r_hat_beyond)),
        "mi_ordering_criterion": mi_ordering_criterion,
        "mi_values": {"Gated": mi_gated, "Partial": mi_partial, "Fixed": mi_fixed},
    }
