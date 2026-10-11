"""Unit tests for image corruption and control transformations."""

import numpy as np
import pytest
from PIL import Image

from src.corruptions.image import black_image, gaussian_blend, patch_mask, swap_blend, swapped


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


def test_swap_blend_s0_returns_img(sample_pil_image, sample_numpy_image):
    """S=0 returns original image exactly for PIL and NumPy inputs."""
    other_pil = Image.new("RGB", (112, 112), color=(0, 255, 0))
    res_pil = swap_blend(sample_pil_image, other_pil, S=0.0)
    assert np.array_equal(np.array(res_pil), np.array(sample_pil_image))

    other_np = np.zeros_like(sample_numpy_image)
    res_np = swap_blend(sample_numpy_image, other_np, S=0.0)
    assert np.array_equal(res_np, sample_numpy_image)


def test_swap_blend_s1_returns_other(sample_pil_image, sample_numpy_image):
    """S=1 returns img_other (resized to img's dimensions if needed)."""
    # Matching dimensions
    other_pil = Image.new("RGB", (112, 112), color=(10, 20, 30))
    res_pil = swap_blend(sample_pil_image, other_pil, S=1.0)
    assert np.array_equal(np.array(res_pil), np.array(other_pil))

    # Differing dimensions
    other_diff = Image.new("RGB", (64, 48), color=(50, 100, 150))
    res_diff = swap_blend(sample_pil_image, other_diff, S=1.0)
    assert isinstance(res_diff, Image.Image)
    assert res_diff.size == sample_pil_image.size
    # Pixels should match the solid color (50, 100, 150)
    arr = np.array(res_diff)
    assert np.all(arr[:, :, 0] == 50)
    assert np.all(arr[:, :, 1] == 100)
    assert np.all(arr[:, :, 2] == 150)


def test_swap_blend_shape_dtype_preserved(sample_pil_image, sample_numpy_image):
    """swap_blend preserves shape, dtype (uint8), and PIL/NumPy format for S in (0, 1)."""
    other_pil = Image.new("RGB", (80, 80), color=(128, 128, 128))
    res_pil = swap_blend(sample_pil_image, other_pil, S=0.4)
    assert isinstance(res_pil, Image.Image)
    assert res_pil.size == sample_pil_image.size

    other_np = np.ones((80, 80, 3), dtype=np.uint8) * 128
    res_np = swap_blend(sample_numpy_image, other_np, S=0.4)
    assert isinstance(res_np, np.ndarray)
    assert res_np.shape == sample_numpy_image.shape
    assert res_np.dtype == np.uint8

