"""Physical sensor corruption operators for Camera and LiDAR in autonomous driving."""

import numpy as np


def apply_camera_fog(
    images: np.ndarray,
    S: float,
    seed: int = 42,
    beta: float = 3.0,
    airlight: float = 0.85,
) -> np.ndarray:
    """Simulate atmospheric fog on camera images using Koschmieder's physical scattering model.

    I_fog(S) = I * exp(-beta * S) + airlight * (1 - exp(-beta * S))

    Parameters
    ----------
    images : np.ndarray of shape (N, 3, H, W) or (3, H, W)
    S : float in [0, 1]
    """
    S_clamped = float(np.clip(S, 0.0, 1.0))
    if S_clamped == 0.0:
        return images.copy()

    transmission = np.exp(-beta * S_clamped)
    fogged = images * transmission + airlight * (1.0 - transmission)
    return np.clip(fogged, 0.0, 1.0).astype(np.float32)


def apply_camera_glare(
    images: np.ndarray,
    S: float,
    seed: int = 42,
) -> np.ndarray:
    """Simulate severe sun glare and sensor saturation washing out camera dynamic range."""
    S_clamped = float(np.clip(S, 0.0, 1.0))
    if S_clamped == 0.0:
        return images.copy()

    rng = np.random.default_rng(seed)
    is_batch = images.ndim == 4
    N = images.shape[0] if is_batch else 1
    H, W = images.shape[-2], images.shape[-1]

    # Radial glare pattern originating from top-center (sun position)
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - W / 2) ** 2 + (y - H * 0.1) ** 2)
    max_d = np.sqrt((W / 2) ** 2 + (H * 0.9) ** 2)
    glare_mask = np.clip(1.0 - (dist / max_d) * 0.8, 0.0, 1.0).astype(np.float32)
    glare_mask = glare_mask[None, :, :]  # (1, H, W)

    if is_batch:
        glare_mask = glare_mask[None, :, :, :]  # (1, 1, H, W)

    # Glare intensity scales with S
    glare_effect = S_clamped * glare_mask * 1.2
    corrupted = images * (1.0 - 0.7 * S_clamped) + glare_effect
    return np.clip(corrupted, 0.0, 1.0).astype(np.float32)


def apply_camera_occlusion(
    images: np.ndarray,
    S: float,
    seed: int = 42,
) -> np.ndarray:
    """Simulate lens dirt/mud/water droplet occlusion on front camera."""
    S_clamped = float(np.clip(S, 0.0, 1.0))
    if S_clamped == 0.0:
        return images.copy()

    rng = np.random.default_rng(seed)
    out = images.copy()
    is_batch = images.ndim == 4
    N = images.shape[0] if is_batch else 1
    H, W = images.shape[-2], images.shape[-1]

    num_splatters = int(1 + S_clamped * 12)
    radius_max = int(4 + S_clamped * 16)

    for i in range(N):
        img_slice = out[i] if is_batch else out
        for _ in range(num_splatters):
            cy = rng.integers(0, H)
            cx = rng.integers(0, W)
            rad = rng.integers(3, max(4, radius_max))
            y, x = np.ogrid[:H, :W]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= rad ** 2
            # Mud brown color (0.35, 0.22, 0.10)
            img_slice[0, mask] = 0.35
            img_slice[1, mask] = 0.22
            img_slice[2, mask] = 0.10

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_lidar_rain(
    lidars: np.ndarray,
    S: float,
    seed: int = 42,
) -> np.ndarray:
    """Simulate LiDAR beam scatter noise and attenuation under heavy rain."""
    S_clamped = float(np.clip(S, 0.0, 1.0))
    if S_clamped == 0.0:
        return lidars.copy()

    rng = np.random.default_rng(seed)
    # 1. Attenuation of valid points
    attenuated = lidars * (1.0 - 0.5 * S_clamped)
    # 2. Random rain clutter returns
    noise_prob = 0.15 * S_clamped
    noise_mask = rng.random(lidars.shape) < noise_prob
    clutter = rng.uniform(0.1, 0.6, size=lidars.shape).astype(np.float32)
    out = np.where(noise_mask, clutter, attenuated)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
