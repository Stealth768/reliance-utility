"""Procedural driving scene generator for paired Camera-LiDAR BEV sensor fusion.

Generates realistic paired sensor representations for autonomous driving:
- Modality A: Front Perspective Camera view (RGB, shape [3, 64, 64])
- Modality B: LiDAR Range / Occupancy grid in Bird's Eye View (shape [1, 32, 32])
- Ground truth target: Road corridor hazard state (4 classes: 0=Clear, 1=Obstacle, 2=Vehicle, 3=Boundary)
"""

import numpy as np
from typing import Dict, Tuple, Optional

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


def generate_driving_scene(
    scene_id: int,
    hazard_class: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single paired Camera RGB view and LiDAR BEV range grid.

    Returns
    -------
    camera_img : np.ndarray of shape (3, 64, 64), float32 in [0, 1]
    lidar_bev : np.ndarray of shape (1, 32, 32), float32 in [0, 1]
    """
    # 1. Perspective Camera View (3, 64, 64)
    # Background: sky on top (rows 0..24), asphalt road on bottom (rows 25..63)
    cam = np.zeros((3, 64, 64), dtype=np.float32)
    # Sky (light blueish gray)
    cam[0, :25, :] = 0.55 + rng.normal(0, 0.02, (25, 64))
    cam[1, :25, :] = 0.65 + rng.normal(0, 0.02, (25, 64))
    cam[2, :25, :] = 0.80 + rng.normal(0, 0.02, (25, 64))
    # Road (dark asphalt gray)
    cam[0, 25:, :] = 0.25 + rng.normal(0, 0.03, (39, 64))
    cam[1, 25:, :] = 0.26 + rng.normal(0, 0.03, (39, 64))
    cam[2, 25:, :] = 0.28 + rng.normal(0, 0.03, (39, 64))
    # Lane markings (white/yellow perspective lines)
    for r in range(25, 64):
        w = int(2 + (r - 25) * 0.4)
        left_col = max(0, 32 - w)
        right_col = min(63, 32 + w)
        cam[0:2, r, max(0, left_col-1):left_col+1] = 0.85
        cam[0:2, r, max(0, right_col-1):right_col+1] = 0.85

    # 2. LiDAR BEV Grid (1, 32, 32)
    # Top-down road corridor: rows 0..31 (y=distance ahead: 0 is closest, 31 is farthest)
    lidar = np.zeros((1, 32, 32), dtype=np.float32)
    # Road ground plane reflection (low intensity)
    lidar[0, :, 10:22] = 0.15 + rng.normal(0, 0.02, (32, 12))

    # 3. Add Hazard Objects according to class
    if hazard_class == 0:
        # Clear path: no central obstacles
        pass
    elif hazard_class == 1:
        # Static Obstacle in lane (small debris/barrier)
        dist_y = rng.integers(10, 22)
        cam_row = 50 - int(dist_y * 1.0)
        cam_col = rng.integers(28, 36)
        color = np.array([0.85, 0.35, 0.1], dtype=np.float32)[:, None, None]
        cam[:, max(0, cam_row-4):cam_row+4, max(0, cam_col-4):cam_col+4] = color
        lidar[0, max(0, dist_y-2):dist_y+2, max(0, cam_col//2-2):cam_col//2+2] = 0.75
    elif hazard_class == 2:
        # Vehicle Ahead (larger rectangular obstacle)
        dist_y = rng.integers(8, 20)
        cam_row = 52 - int(dist_y * 1.1)
        cam_col = rng.integers(26, 38)
        color = np.array([0.9, 0.15, 0.15], dtype=np.float32)[:, None, None]
        cam[:, max(0, cam_row-8):cam_row+4, max(0, cam_col-8):cam_col+8] = color
        lidar[0, max(0, dist_y-3):dist_y+3, max(0, cam_col//2-3):cam_col//2+3] = 0.95
    elif hazard_class == 3:
        # Road Boundary Hazard (guardrail / ditch intrusion on flank)
        dist_y = rng.integers(12, 25)
        side = rng.choice([-1, 1])
        cam_col = 32 + side * rng.integers(14, 22)
        cam_row = 50 - int(dist_y * 0.9)
        color = np.array([0.8, 0.8, 0.2], dtype=np.float32)[:, None, None]
        cam[:, max(0, cam_row-6):cam_row+4, max(0, cam_col-5):cam_col+5] = color
        lidar_col = 16 + side * rng.integers(8, 13)
        lidar[0, max(0, dist_y-4):dist_y+4, max(0, lidar_col-2):lidar_col+2] = 0.85

    cam = np.clip(cam, 0.0, 1.0)
    lidar = np.clip(lidar, 0.0, 1.0)
    return cam, lidar


def generate_bev_dataset(
    n_samples: int = 4000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate a balanced synthetic Camera-LiDAR dataset for BEV hazard classification."""
    rng = np.random.default_rng(seed)
    n_classes = 4

    cameras = np.zeros((n_samples, 3, 64, 64), dtype=np.float32)
    lidars = np.zeros((n_samples, 1, 32, 32), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        cls = i % n_classes
        labels[i] = cls
        cam, lid = generate_driving_scene(i, cls, rng)
        cameras[i] = cam
        lidars[i] = lid

    return {
        "cameras": cameras,
        "lidars": lidars,
        "labels": labels,
    }


def split_bev_data(
    data: Dict[str, np.ndarray],
    n_train: int = 3000,
    n_val: int = 500,
    n_test: int = 500,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split dataset deterministically into train, val, and test subsets."""
    train = {k: v[:n_train] for k, v in data.items()}
    val = {k: v[n_train: n_train + n_val] for k, v in data.items()}
    test = {k: v[n_train + n_val: n_train + n_val + n_test] for k, v in data.items()}
    return train, val, test


class BEVDataset(_BaseDataset):
    """PyTorch Dataset wrapper for paired Camera-LiDAR scenes."""

    def __init__(
        self,
        cameras: np.ndarray,
        lidars: np.ndarray,
        labels: np.ndarray,
        camera_dropout_p: float = 0.0,
        seed: int = 0,
    ):
        self.cameras = cameras
        self.lidars = lidars
        self.labels = labels
        self.camera_dropout_p = float(camera_dropout_p)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        cam = self.cameras[idx]
        lid = self.lidars[idx]
        lbl = self.labels[idx]
        cam_removed = bool(self.camera_dropout_p > 0 and self.rng.random() < self.camera_dropout_p)

        if TORCH_AVAILABLE:
            return (
                torch.from_numpy(cam),
                torch.from_numpy(lid),
                torch.tensor(lbl, dtype=torch.long),
                torch.tensor(cam_removed, dtype=torch.bool),
            )
        return cam, lid, lbl, cam_removed
