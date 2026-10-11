"""Image corruptions and control transformations for VLM reliance-utility experiments.

Pure NumPy and PIL implementations:
- gaussian_blend: (1-S)*img + S*noise, noise ~ N(mean=128, std=matching image std), clipped [0, 255]
- patch_mask: mask fraction S of 28x28 patches with gray 128
- swapped: control transformation replacing question's image with another image
- black_image: all-zeros image matching given size
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image


def _to_numpy_uint8(img: Union[Image.Image, np.ndarray]) -> Tuple[np.ndarray, bool]:
    """Convert input image to uint8 RGB numpy array and record if it was PIL."""
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return arr, True
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr, False


def _to_output(arr: np.ndarray, return_pil: bool) -> Union[Image.Image, np.ndarray]:
    """Convert numpy array back to PIL Image if input was PIL Image."""
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    if return_pil:
        return Image.fromarray(arr_uint8)
    return arr_uint8


def gaussian_blend(
    img: Union[Image.Image, np.ndarray],
    S: float,
    seed: int,
) -> Union[Image.Image, np.ndarray]:
    """Blend image with Gaussian noise: (1 - S) * img + S * noise.

    Noise is drawn from N(mean=128, std=std(img)), clipped to [0, 255], uint8.
    For S=0, returns the exact original image.

    Parameters
    ----------
    img : PIL.Image or np.ndarray
        Input image (RGB).
    S : float
        Corruption severity in [0, 1].
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    PIL.Image or np.ndarray
        Corrupted image of same type and shape.
    """
    arr, is_pil = _to_numpy_uint8(img)
    if S <= 0.0:
        return img if is_pil else arr.copy()

    S_clamped = min(max(float(S), 0.0), 1.0)
    rng = np.random.default_rng(seed)

    # Compute standard deviation of input image (fallback to 30.0 if image is constant)
    img_std = float(np.std(arr))
    if img_std < 1e-4:
        img_std = 30.0

    # Draw Gaussian noise matching image std around mean 128
    noise = rng.normal(loc=128.0, scale=img_std, size=arr.shape)
    noise = np.clip(noise, 0.0, 255.0)

    # Linear blend
    blended = (1.0 - S_clamped) * arr.astype(float) + S_clamped * noise
    return _to_output(blended, is_pil)


def patch_mask(
    img: Union[Image.Image, np.ndarray],
    S: float,
    seed: int,
    patch: int = 28,
) -> Union[Image.Image, np.ndarray]:
    """Mask a fraction S of non-overlapping patch x patch blocks with gray 128.

    Patches are selected randomly without replacement, seeded by seed.
    For S=0, returns the exact original image.
    For S=1, masks all patches in the grid.

    Parameters
    ----------
    img : PIL.Image or np.ndarray
        Input image (RGB).
    S : float
        Fraction of patches to mask, in [0, 1].
    seed : int
        RNG seed for patch selection.
    patch : int, default=28
        Patch size in pixels (28 aligns with Qwen2-VL patch merging).

    Returns
    -------
    PIL.Image or np.ndarray
        Masked image of same type and shape.
    """
    arr, is_pil = _to_numpy_uint8(img)
    if S <= 0.0:
        return img if is_pil else arr.copy()

    S_clamped = min(max(float(S), 0.0), 1.0)
    H, W = arr.shape[:2]

    # Grid of patch top-left coordinates
    n_rows = (H + patch - 1) // patch
    n_cols = (W + patch - 1) // patch
    total_patches = n_rows * n_cols

    if total_patches == 0:
        return img if is_pil else arr.copy()

    # Number of patches to mask
    if S_clamped >= 1.0:
        k_mask = total_patches
    else:
        k_mask = int(round(S_clamped * total_patches))

    out_arr = arr.copy()
    if k_mask == 0:
        return _to_output(out_arr, is_pil)

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(total_patches, size=k_mask, replace=False)

    for idx in chosen_indices:
        r = (idx // n_cols) * patch
        c = (idx % n_cols) * patch
        r_end = min(r + patch, H)
        c_end = min(c + patch, W)
        out_arr[r:r_end, c:c_end] = 128

    return _to_output(out_arr, is_pil)


def swapped(img_other: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
    """Return another question's image as an uncorrupted cross-modal mismatch control.

    Parameters
    ----------
    img_other : PIL.Image or np.ndarray
        The alternative image to substitute.

    Returns
    -------
    PIL.Image or np.ndarray
        Copy/identity of the swapped image.
    """
    if isinstance(img_other, Image.Image):
        return img_other.copy()
    return np.asarray(img_other).copy()


def black_image(size: Tuple[int, int]) -> Image.Image:
    """Create an all-black (zeros) RGB image of the specified size (width, height).

    Parameters
    ----------
    size : Tuple[int, int]
        Image size as (width, height).

    Returns
    -------
    PIL.Image.Image
        All-black RGB PIL Image.
    """
    width, height = size
    return Image.new("RGB", (int(width), int(height)), color=(0, 0, 0))


def swap_blend(
    img: Union[Image.Image, np.ndarray],
    img_other: Union[Image.Image, np.ndarray],
    S: float,
) -> Union[Image.Image, np.ndarray]:
    """Blend image with another question's image: (1 - S) * img + S * img_other.

    img_other is resized to match img's spatial resolution if needed.
    Returns uint8 array or PIL Image preserving input type and shape.
    For S=0, returns the exact original image.
    For S=1, returns img_other resized to img's dimensions.

    Parameters
    ----------
    img : PIL.Image or np.ndarray
        Original image.
    img_other : PIL.Image or np.ndarray
        Donor image from another question.
    S : float
        Blend severity parameter in [0, 1].

    Returns
    -------
    PIL.Image or np.ndarray
        Blended image preserving img's format, shape, and uint8 dtype.
    """
    arr, is_pil = _to_numpy_uint8(img)
    if S <= 0.0:
        return img if is_pil else arr.copy()

    S_clamped = min(max(float(S), 0.0), 1.0)
    H, W = arr.shape[:2]

    # Convert img_other and resize if spatial dimensions differ
    arr_other, _ = _to_numpy_uint8(img_other)
    if arr_other.shape[:2] != (H, W):
        pil_other = Image.fromarray(arr_other)
        pil_other = pil_other.resize((W, H), Image.Resampling.BILINEAR)
        arr_other = np.array(pil_other, dtype=np.uint8)

    if S_clamped >= 1.0:
        return _to_output(arr_other, is_pil)

    # Linear blend: (1 - S) * img + S * img_other
    blended = (1.0 - S_clamped) * arr.astype(float) + S_clamped * arr_other.astype(float)
    return _to_output(blended, is_pil)

