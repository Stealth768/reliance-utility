"""CPU unit tests for BEV Camera-LiDAR sensor fusion module."""

import numpy as np
import pytest

from src.bev.corruptions import (
    apply_camera_fog,
    apply_camera_glare,
    apply_camera_occlusion,
    apply_lidar_rain,
)
from src.bev.data import generate_bev_dataset, split_bev_data, BEVDataset
from src.bev.model import TORCH_AVAILABLE, BEVFixedFusionModel, BEVGatedFusionModel


def test_bev_dataset_generation():
    """generate_bev_dataset produces consistent shapes and deterministic labels."""
    data = generate_bev_dataset(n_samples=40, seed=42)
    assert data["cameras"].shape == (40, 3, 64, 64)
    assert data["lidars"].shape == (40, 1, 32, 32)
    assert data["labels"].shape == (40,)
    assert set(np.unique(data["labels"])) == {0, 1, 2, 3}

    train, val, test = split_bev_data(data, n_train=20, n_val=10, n_test=10)
    assert len(train["labels"]) == 20
    assert len(val["labels"]) == 10
    assert len(test["labels"]) == 10


def test_bev_corruptions_shape_and_bounds():
    """Sensor corruptions preserve tensor dimensions and remain bounded in [0, 1]."""
    cam = np.ones((4, 3, 64, 64), dtype=np.float32) * 0.5
    lid = np.ones((4, 1, 32, 32), dtype=np.float32) * 0.5

    fog_0 = apply_camera_fog(cam, S=0.0)
    fog_1 = apply_camera_fog(cam, S=1.0)
    assert np.array_equal(fog_0, cam)
    assert fog_1.shape == cam.shape
    assert np.all(fog_1 >= 0.0) and np.all(fog_1 <= 1.0)
    assert not np.array_equal(fog_1, cam)

    glare_1 = apply_camera_glare(cam, S=1.0)
    assert glare_1.shape == cam.shape
    assert np.all(glare_1 >= 0.0) and np.all(glare_1 <= 1.0)

    occ_1 = apply_camera_occlusion(cam, S=1.0)
    assert occ_1.shape == cam.shape
    assert np.all(occ_1 >= 0.0) and np.all(occ_1 <= 1.0)

    rain_1 = apply_lidar_rain(lid, S=1.0)
    assert rain_1.shape == lid.shape
    assert np.all(rain_1 >= 0.0) and np.all(rain_1 <= 1.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for BEV model test")
def test_bev_models_forward():
    """BEV models run forward pass and respect camera removal null token."""
    import torch

    fixed = BEVFixedFusionModel(hidden_dim=32, num_classes=4)
    gated = BEVGatedFusionModel(hidden_dim=32, num_classes=4)

    cam1 = torch.randn(2, 3, 64, 64)
    cam2 = torch.randn(2, 3, 64, 64)
    lid = torch.randn(2, 1, 32, 32)

    # 1. Fixed model forward
    out_fix, gate_fix = fixed(cam1, lid)
    assert out_fix.shape == (2, 4)
    assert gate_fix.shape == (2, 1)

    # 2. Gated model forward
    out_gat, gate_gat = gated(cam1, lid)
    assert out_gat.shape == (2, 4)
    assert gate_gat.shape == (2, 1)
    assert torch.all(gate_gat >= 0.0) and torch.all(gate_gat <= 1.0)

    # 3. Null token test: when cam_removed=True, cam1 and cam2 must yield identical outputs
    out_null1, _ = gated(cam1, lid, cam_removed=True)
    out_null2, _ = gated(cam2, lid, cam_removed=True)
    assert torch.allclose(out_null1, out_null2, atol=1e-5)
