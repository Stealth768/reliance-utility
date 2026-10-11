"""Dataset generators and splits for modality-dropout fusion case study.

Provides:
1. SyntheticTwoViewDataset: higher-dimension synthetic benchmark with overlapping classes.
2. FashionMNISTTwoViewDataset: two-view proxy using Fashion-MNIST split horizontally
   (modality A = left 14 cols, modality B = right 14 cols).
   NOTE: This is strictly a two-view proxy, NOT a true multimodal benchmark.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
    _BaseDataset = TorchDataset
except ImportError:
    torch = None
    TorchDataset = object
    TORCH_AVAILABLE = False
    _BaseDataset = object


def generate_synthetic_twoview_data(
    n_samples: int = 7000,
    dim_a: int = 64,
    dim_b: int = 64,
    n_classes: int = 4,
    class_overlap: float = 0.3,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic two-view data with controlled class centroids.

    Splits: 5000 train, 1000 val, 1000 test (when n_samples=7000).
    """
    rng = np.random.default_rng(seed)

    # Centroids per class
    centroids_a = rng.standard_normal((n_classes, dim_a)) * (1.0 - class_overlap)
    centroids_b = rng.standard_normal((n_classes, dim_b)) * (1.0 - class_overlap)

    labels = rng.integers(0, n_classes, size=n_samples)
    x_a = centroids_a[labels] + rng.standard_normal((n_samples, dim_a)) * 0.5
    x_b = centroids_b[labels] + rng.standard_normal((n_samples, dim_b)) * 0.5

    return {
        "x_a": x_a.astype(np.float32),
        "x_b": x_b.astype(np.float32),
        "labels": labels.astype(np.int64),
    }


def split_data(
    data: Dict[str, np.ndarray],
    n_train: int = 5000,
    n_val: int = 1000,
    n_test: int = 1000,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Deterministically slice data into train, val, test subsets."""
    train = {k: v[:n_train] for k, v in data.items()}
    val = {k: v[n_train: n_train + n_val] for k, v in data.items()}
    test = {k: v[n_train + n_val: n_train + n_val + n_test] for k, v in data.items()}
    return train, val, test


def apply_modality_a_degradation(
    x_a: np.ndarray,
    S: float,
    seed: int,
) -> np.ndarray:
    """Degrade modality A via linear interpolation toward standard Gaussian noise."""
    S_clamped = float(np.clip(S, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x_a.shape).astype(np.float32)
    return (1.0 - S_clamped) * x_a + S_clamped * noise


class SyntheticTwoViewDataset(_BaseDataset):
    """PyTorch Dataset wrapper for synthetic two-view data."""

    def __init__(
        self,
        data_dict: Dict[str, np.ndarray],
        dropout_p: float = 0.0,
        seed: int = 0,
    ):
        self.x_a = data_dict["x_a"]
        self.x_b = data_dict["x_b"]
        self.labels = data_dict["labels"]
        self.dropout_p = float(dropout_p)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        a_removed = bool(self.dropout_p > 0 and self.rng.random() < self.dropout_p)
        xa = self.x_a[idx]
        xb = self.x_b[idx]
        lbl = self.labels[idx]

        if TORCH_AVAILABLE:
            return (
                torch.from_numpy(xa),
                torch.from_numpy(xb),
                torch.tensor(lbl, dtype=torch.long),
                torch.tensor(a_removed, dtype=torch.bool),
            )
        return xa, xb, lbl, a_removed


class FashionMNISTTwoViewDataset(_BaseDataset):
    """Two-view proxy dataset created from Fashion-MNIST.

    Modality A = left 14 columns of the 28x28 image (flattened to 392-d).
    Modality B = right 14 columns of the 28x28 image (flattened to 392-d).
    NOTE: Strictly a two-view proxy for methodological validation, not a true multimodal dataset.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        dropout_p: float = 0.0,
        seed: int = 0,
    ):
        # images expected as (N, 28, 28) uint8 or float32 in [0, 1]
        self.x_a = images[:, :, :14].reshape(len(images), -1).astype(np.float32)
        self.x_b = images[:, :, 14:].reshape(len(images), -1).astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.dropout_p = float(dropout_p)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        a_removed = bool(self.dropout_p > 0 and self.rng.random() < self.dropout_p)
        xa = self.x_a[idx]
        xb = self.x_b[idx]
        lbl = self.labels[idx]

        if TORCH_AVAILABLE:
            return (
                torch.from_numpy(xa),
                torch.from_numpy(xb),
                torch.tensor(lbl, dtype=torch.long),
                torch.tensor(a_removed, dtype=torch.bool),
            )
        return xa, xb, lbl, a_removed
