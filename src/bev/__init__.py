"""BEV sensor fusion package for Camera-LiDAR reliance-utility case study."""

from src.bev.corruptions import (
    apply_camera_fog,
    apply_camera_glare,
    apply_camera_occlusion,
    apply_lidar_rain,
)
from src.bev.data import (
    generate_bev_dataset,
    split_bev_data,
    BEVDataset,
)
from src.bev.model import (
    BEVFixedFusionModel,
    BEVGatedFusionModel,
)

__all__ = [
    "apply_camera_fog",
    "apply_camera_glare",
    "apply_camera_occlusion",
    "apply_lidar_rain",
    "generate_bev_dataset",
    "split_bev_data",
    "BEVDataset",
    "BEVFixedFusionModel",
    "BEVGatedFusionModel",
]
