"""Unit tests for image corruption and control transformations."""

import numpy as np
import pytest
from PIL import Image

from src.corruptions.image import black_image, gaussian_blend, patch_mask, swapped


@pytest.fixture
def sample_pil_image():
    """Create a synthetic PIL RGB image with realistic variance."""
    rng = np.random.default_rng(42)
    arr = rng.integers(20, 240, size=(112, 112, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def sample_numpy_image():
    """Create a synthetic NumPy RGB array with realistic variance."""
    rng = np.random.default_rng(42)
    return rng.integers(20, 240, size=(112, 112, 3), dtype=np.uint8)


def test_gaussian_blend_s0_exact(sample_pil_image, sample_numpy_image):
    """S=0 returns the original image exactly for both PIL and NumPy inputs."""
    # PIL
    out_pil = gaussian_blend(sample_pil_image, S=0.0, seed=123)
    assert np.array_equal(np.array(out_pil), np.array(sample_pil_image))

    # NumPy
    out_np = gaussian_blend(sample_numpy_image, S=0.0, seed=123)
    assert np.array_equal(out_np, sample_numpy_image)


def test_patch_mask_s0_exact(sample_pil_image, sample_numpy_image):
    """S=0 returns the original image exactly for patch_mask."""
    out_pil = patch_mask(sample_pil_image, S=0.0, seed=123)
    assert np.array_equal(np.array(out_pil), np.array(sample_pil_image))

    out_np = patch_mask(sample_numpy_image, S=0.0, seed=123)
    assert np.array_equal(out_np, sample_numpy_image)


def test_patch_mask_s1_masks_all(sample_pil_image):
    """S=1 for patch_mask masks all patches (entire image becomes gray 128)."""
    # 112x112 is a multiple of 28 (4x4 = 16 patches)
    out = patch_mask(sample_pil_image, S=1.0, seed=42, patch=28)
    arr = np.array(out)
    assert np.all(arr == 128), "Expected all pixels to be masked with gray 128 when S=1.0"


def test_reproducibility_same_seed(sample_pil_image):
    """Same seed gives identical output for both corruptions."""
    # Gaussian blend
    gb1 = gaussian_blend(sample_pil_image, S=0.5, seed=777)
    gb2 = gaussian_blend(sample_pil_image, S=0.5, seed=777)
    assert np.array_equal(np.array(gb1), np.array(gb2))

    # Patch mask
    pm1 = patch_mask(sample_pil_image, S=0.5, seed=777, patch=28)
    pm2 = patch_mask(sample_pil_image, S=0.5, seed=777, patch=28)
    assert np.array_equal(np.array(pm1), np.array(pm2))


def test_output_shape_and_dtype_preserved(sample_pil_image, sample_numpy_image):
    """Output shape and dtype (uint8) are preserved."""
    # PIL input returns PIL with matching dimensions
    out_pil = gaussian_blend(sample_pil_image, S=0.75, seed=99)
    assert isinstance(out_pil, Image.Image)
    assert out_pil.size == sample_pil_image.size

    # NumPy input returns NumPy uint8 with matching shape
    out_np = gaussian_blend(sample_numpy_image, S=0.75, seed=99)
    assert isinstance(out_np, np.ndarray)
    assert out_np.shape == sample_numpy_image.shape
    assert out_np.dtype == np.uint8

    out_pm = patch_mask(sample_numpy_image, S=0.75, seed=99)
    assert isinstance(out_pm, np.ndarray)
    assert out_pm.shape == sample_numpy_image.shape
    assert out_pm.dtype == np.uint8


def test_swapped_control(sample_pil_image):
    """Swapped control returns copy of alternative image."""
    other = Image.new("RGB", (64, 64), color=(255, 0, 0))
    swp = swapped(other)
    assert swp.size == (64, 64)
    assert np.array_equal(np.array(swp), np.array(other))


def test_black_image():
    """black_image produces all-zero RGB image of requested size."""
    blk = black_image((100, 80))
    assert isinstance(blk, Image.Image)
    assert blk.size == (100, 80)
    arr = np.array(blk)
    assert arr.shape == (80, 100, 3)
    assert np.all(arr == 0)
