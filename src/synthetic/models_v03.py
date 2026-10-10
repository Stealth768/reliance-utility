"""Models for synthetic benchmark iteration v0.3.

Provides:
- OracleGatedModel: Idealized oracle positive control model that explicitly receives
  severity S and applies gate 1[S < 0.65] to modality A. Not a realistic model.
- SharpGatedModel: Quality-aware fusion model with scaled sigmoid steepness
  (gamma * gamma_scale), calibrated on the validation set.
"""

from typing import Optional, Union
import numpy as np
from scipy.special import expit, softmax

from src.synthetic.models import FixedModel, GatedModel, PartialGatedModel


class OracleGatedModel(GatedModel):
    """Oracle positive control model with explicit degradation severity awareness.

    CRITICAL NOTE: This model is an idealized oracle positive control, NOT a realistic
    multimodal model. It receives the true degradation severity parameter S explicitly
    and shuts off modality A whenever S >= cutoff (default cutoff = 0.65):
    weight_A = w_A * 1[S < 0.65].
    """

    def __init__(
        self,
        w_A: float = 1.0,
        w_B: float = 1.0,
        cutoff: float = 0.65,
    ):
        super().__init__(w_A=w_A, w_B=w_B)
        self.cutoff = float(cutoff)
        self._current_S: Optional[float] = None

    def set_severity(self, S: Optional[float]) -> "OracleGatedModel":
        """Set current degradation severity S for subsequent predictions."""
        self._current_S = float(S) if S is not None else None
        return self

    def calibrate_gate(
        self,
        X_A_val: np.ndarray,
        seed: int = 42,
    ) -> "OracleGatedModel":
        """No-op gate calibration for oracle model (preserves runner interface)."""
        super().calibrate_gate(X_A_val, seed=seed)
        return self

    def compute_logits(
        self,
        X_A: Optional[np.ndarray],
        X_B: np.ndarray,
        S: Optional[float] = None,
    ) -> np.ndarray:
        """Compute logits: 1[S < 0.65] * w_A * l_A(X_A) + w_B * l_B(X_B)."""
        n_samples = len(X_B)
        l_B = self.compute_logits_B(X_B)

        if X_A is None:
            return self.w_B * l_B

        s_val = S if S is not None else self._current_S
        oracle_gate = 1.0 if (s_val is None or s_val < self.cutoff) else 0.0

        l_A = self.compute_logits_A(X_A, n_samples)
        return oracle_gate * self.w_A * l_A + self.w_B * l_B

    def predict_proba(
        self,
        X_A: Optional[np.ndarray],
        X_B: np.ndarray,
        S: Optional[float] = None,
    ) -> np.ndarray:
        """Output class probability distributions via softmax."""
        logits = self.compute_logits(X_A, X_B, S=S)
        return softmax(logits, axis=-1)

    def predict(
        self,
        X_A: Optional[np.ndarray],
        X_B: np.ndarray,
        S: Optional[float] = None,
    ) -> np.ndarray:
        """Output predicted class indices."""
        proba = self.predict_proba(X_A, X_B, S=S)
        return np.argmax(proba, axis=-1)


class SharpGatedModel(GatedModel):
    """Quality-aware multimodal model with scaled sigmoid steepness.

    Steepness parameter gamma is scaled by gamma_scale: gamma_scaled = gamma * gamma_scale.
    The midpoint is calibrated on the validation set using the exact same procedure as GatedModel.
    """

    def __init__(
        self,
        gamma_scale: float = 1.0,
        w_A: float = 1.0,
        w_B: float = 1.0,
    ):
        super().__init__(w_A=w_A, w_B=w_B)
        self.gamma_scale = float(gamma_scale)

    def calibrate_gate(
        self,
        X_A_val: np.ndarray,
        seed: int = 42,
    ) -> "SharpGatedModel":
        """Calibrate midpoint and scale steepness gamma by gamma_scale."""
        super().calibrate_gate(X_A_val, seed=seed)
        self.gamma = self.gamma * self.gamma_scale
        return self
