"""Toy multimodal fusion models for synthetic reliance-utility validation."""

from typing import Optional, Tuple, Union
import numpy as np
from scipy.special import expit, softmax

from src.synthetic.data import degrade_A


class FixedModel:
    """Quality-blind multimodal fusion model with fixed logit weights.

    Fits per-modality linear/prototype distance logit functions l_A, l_B on clean
    training data only. Weights w_A, w_B are constant and blind to input corruption.
    When X_A is None, the modality A contribution is zero.
    """

    def __init__(self, w_A: float = 1.0, w_B: float = 1.0):
        self.w_A = float(w_A)
        self.w_B = float(w_B)
        self.proto_A: Optional[np.ndarray] = None
        self.proto_B: Optional[np.ndarray] = None
        self.sigma_A: float = 1.0
        self.sigma_B: float = 1.0
        self.k: int = 4
        self.d: int = 8
        self.tau: float = 1.0

    def fit(
        self,
        X_A_train: np.ndarray,
        X_B_train: np.ndarray,
        y_train: np.ndarray,
        k: Optional[int] = None,
    ) -> "FixedModel":
        """Fit prototype centroids and variance scales on clean training data.

        Parameters
        ----------
        X_A_train : np.ndarray
            Clean modality A features, shape (n, d).
        X_B_train : np.ndarray
            Clean modality B features, shape (n, d).
        y_train : np.ndarray
            Class labels, shape (n,).
        k : Optional[int], default=None
            Number of classes (inferred from y_train if not provided).

        Returns
        -------
        FixedModel
            Fitted instance.
        """
        if k is None:
            self.k = int(np.max(y_train) + 1)
        else:
            self.k = int(k)

        self.d = X_A_train.shape[1]
        self.tau = float(np.std(X_A_train))

        # Compute class-conditional prototype centroids
        self.proto_A = np.zeros((self.k, self.d), dtype=float)
        self.proto_B = np.zeros((self.k, self.d), dtype=float)

        var_A_accum = 0.0
        var_B_accum = 0.0

        for c in range(self.k):
            mask = y_train == c
            if np.any(mask):
                self.proto_A[c] = np.mean(X_A_train[mask], axis=0)
                self.proto_B[c] = np.mean(X_B_train[mask], axis=0)
                var_A_accum += float(np.sum((X_A_train[mask] - self.proto_A[c]) ** 2))
                var_B_accum += float(np.sum((X_B_train[mask] - self.proto_B[c]) ** 2))

        n_total = len(y_train) * self.d
        self.sigma_A = float(np.sqrt(max(var_A_accum / n_total, 1e-4)))
        self.sigma_B = float(np.sqrt(max(var_B_accum / n_total, 1e-4)))

        return self

    def compute_logits_A(self, X_A: Optional[np.ndarray], n_samples: int) -> np.ndarray:
        """Compute logit contribution from modality A."""
        if X_A is None:
            return np.zeros((n_samples, self.k), dtype=float)

        # Distance to class prototypes: - ||x - proto||^2 / (2 * sigma_A^2)
        diff = X_A[:, None, :] - self.proto_A[None, :, :]
        sq_dist = np.sum(diff**2, axis=-1)
        return -sq_dist / (2.0 * self.sigma_A**2)

    def compute_logits_B(self, X_B: np.ndarray) -> np.ndarray:
        """Compute logit contribution from modality B."""
        diff = X_B[:, None, :] - self.proto_B[None, :, :]
        sq_dist = np.sum(diff**2, axis=-1)
        return -sq_dist / (2.0 * self.sigma_B**2)

    def compute_logits(
        self,
        X_A: Optional[np.ndarray],
        X_B: np.ndarray,
    ) -> np.ndarray:
        """Compute fused logits w_A * l_A(X_A) + w_B * l_B(X_B)."""
        n_samples = len(X_B)
        l_B = self.compute_logits_B(X_B)
        l_A = self.compute_logits_A(X_A, n_samples)
        return self.w_A * l_A + self.w_B * l_B

    def predict_proba(
        self,
        X_A: Optional[np.ndarray],
        X_B: np.ndarray,
    ) -> np.ndarray:
        """Output class probability distributions via softmax."""
        logits = self.compute_logits(X_A, X_B)
        return softmax(logits, axis=-1)

    def predict(
        self,
        X_A: Optional[np.ndarray],
        X_B: np.ndarray,
    ) -> np.ndarray:
        """Output predicted class indices."""
        proba = self.predict_proba(X_A, X_B)
        return np.argmax(proba, axis=-1)


class GatedModel(FixedModel):
    """Quality-aware multimodal fusion model with input-dependent gating.

    Calculates gate g in [0, 1] from X_A alone (distance to nearest class prototype
    passed through a calibrated sigmoid). As degradation increases, g drops to 0,
    shedding reliance on corrupted modality A.
    """

    def __init__(self, w_A: float = 1.0, w_B: float = 1.0):
        super().__init__(w_A=w_A, w_B=w_B)
        self.midpoint: float = 2.0
        self.gamma: float = 4.0

    def calibrate_gate(
        self,
        X_A_val: np.ndarray,
        seed: int = 42,
    ) -> "GatedModel":
        """Calibrate sigmoid midpoint and steepness on a separate validation set.

        Parameters
        ----------
        X_A_val : np.ndarray
            Clean validation features for modality A.
        seed : int, default=42
            Seed for synthesizing validation noise.

        Returns
        -------
        GatedModel
            Calibrated model instance.
        """
        # Distance to nearest prototype on clean validation set
        d_clean = np.min(
            np.linalg.norm(X_A_val[:, None, :] - self.proto_A[None, :, :], axis=-1),
            axis=1,
        )

        # Distance to nearest prototype on pure noise validation set (S=1)
        rng = np.random.default_rng(seed)
        noise_val = rng.normal(0.0, self.tau, size=X_A_val.shape)
        d_noise = np.min(
            np.linalg.norm(noise_val[:, None, :] - self.proto_A[None, :, :], axis=-1),
            axis=1,
        )

        # Midpoint is set halfway between clean and noise regimes
        mean_clean = float(np.mean(d_clean))
        mean_noise = float(np.mean(d_noise))

        self.midpoint = (mean_clean + mean_noise) / 2.0
        self.gamma = 6.0 / max(mean_noise - mean_clean, 1e-4)

        return self

    def compute_gate(self, X_A: Optional[np.ndarray]) -> np.ndarray:
        """Compute reliability gate g in [0, 1] from X_A only."""
        if X_A is None:
            return np.array([0.0])

        d_nearest = np.min(
            np.linalg.norm(X_A[:, None, :] - self.proto_A[None, :, :], axis=-1),
            axis=1,
        )
        # Sigmoid: g drops as d_nearest exceeds midpoint
        return expit(-self.gamma * (d_nearest - self.midpoint))

    def compute_logits(
        self,
        X_A: Optional[np.ndarray],
        X_B: np.ndarray,
    ) -> np.ndarray:
        """Compute dynamically gated logits: g * w_A * l_A(X_A) + w_B * l_B(X_B)."""
        n_samples = len(X_B)
        l_B = self.compute_logits_B(X_B)

        if X_A is None:
            return self.w_B * l_B

        g = self.compute_gate(X_A)[:, None]
        l_A = self.compute_logits_A(X_A, n_samples)
        return g * self.w_A * l_A + self.w_B * l_B


class PartialGatedModel(GatedModel):
    """Partially quality-aware model with restricted dynamic gating range.

    Gate is bounded to [0.5, 1.0]: gate = 0.5 * g + 0.5.
    Demonstrates that the Misalignment Index monotonically orders models by awareness.
    """

    def compute_logits(
        self,
        X_A: Optional[np.ndarray],
        X_B: np.ndarray,
    ) -> np.ndarray:
        """Compute partially gated logits: (0.5 * g + 0.5) * w_A * l_A(X_A) + w_B * l_B(X_B)."""
        n_samples = len(X_B)
        l_B = self.compute_logits_B(X_B)

        if X_A is None:
            return self.w_B * l_B

        g = self.compute_gate(X_A)[:, None]
        g_partial = 0.5 * g + 0.5
        l_A = self.compute_logits_A(X_A, n_samples)
        return g_partial * self.w_A * l_A + self.w_B * l_B
