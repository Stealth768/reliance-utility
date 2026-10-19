"""Camera-LiDAR Bird's Eye View (BEV) fusion architectures for autonomous driving."""

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    _BaseModule = nn.Module
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    _BaseModule = object


class CameraEncoder(_BaseModule):
    """Convolutional perspective camera encoder."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> (16, 32, 32)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 8, 8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.conv(x)


class LiDAREncoder(_BaseModule):
    """Convolutional top-down BEV LiDAR range grid encoder."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (16, 16, 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 4, 4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.conv(x)


class BEVFixedFusionModel(_BaseModule):
    """Quality-blind Camera-LiDAR BEV fusion model (Fixed Fusion).

    Fuses camera and LiDAR feature representations using a static concatenation head
    without dynamic reliability gating.
    """

    def __init__(self, hidden_dim: int = 64, num_classes: int = 4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for BEVFixedFusionModel.")
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.cam_enc = CameraEncoder(hidden_dim)
        self.lid_enc = LiDAREncoder(hidden_dim)

        # Learned null token for Camera when removed
        self.null_cam = nn.Parameter(torch.zeros(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        cameras: "torch.Tensor",
        lidars: "torch.Tensor",
        cam_removed: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        batch_size = cameras.shape[0]

        h_cam_raw = self.cam_enc(cameras)
        h_lid = self.lid_enc(lidars)

        if cam_removed is not None:
            if isinstance(cam_removed, bool):
                mask = torch.full((batch_size, 1), cam_removed, device=cameras.device, dtype=torch.bool)
            else:
                mask = cam_removed.view(batch_size, 1)
            null_exp = self.null_cam.unsqueeze(0).expand(batch_size, -1)
            h_cam = torch.where(mask, null_exp, h_cam_raw)
        else:
            h_cam = h_cam_raw

        fused = torch.cat([h_cam, h_lid], dim=-1)
        logits = self.head(fused)
        # Fixed model gives constant 0.5 gate weight
        gate = torch.full((batch_size, 1), 0.5, device=cameras.device)
        return logits, gate


class BEVGatedFusionModel(_BaseModule):
    """Quality-aware Camera-LiDAR BEV fusion model with dynamic reliability gating.

    Computes a learned gate g in [0, 1] conditioned on LiDAR feature agreement:
        g = sigmoid(linear(h_lidar))
        logits = g * head_cam(h_cam) + (1 - g) * head_lid(h_lidar)
    """

    def __init__(self, hidden_dim: int = 64, num_classes: int = 4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for BEVGatedFusionModel.")
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.cam_enc = CameraEncoder(hidden_dim)
        self.lid_enc = LiDAREncoder(hidden_dim)

        self.null_cam = nn.Parameter(torch.zeros(hidden_dim))

        self.head_cam = nn.Linear(hidden_dim, num_classes)
        self.head_lid = nn.Linear(hidden_dim, num_classes)

        # Gate on camera stream computed from LiDAR representation
        self.gate_fc = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        cameras: "torch.Tensor",
        lidars: "torch.Tensor",
        cam_removed: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        batch_size = cameras.shape[0]

        h_cam_raw = self.cam_enc(cameras)
        h_lid = self.lid_enc(lidars)

        if cam_removed is not None:
            if isinstance(cam_removed, bool):
                mask = torch.full((batch_size, 1), cam_removed, device=cameras.device, dtype=torch.bool)
            else:
                mask = cam_removed.view(batch_size, 1)
            null_exp = self.null_cam.unsqueeze(0).expand(batch_size, -1)
            h_cam = torch.where(mask, null_exp, h_cam_raw)
        else:
            h_cam = h_cam_raw

        logits_cam = self.head_cam(h_cam)
        logits_lid = self.head_lid(h_lid)

        # Dynamic reliability gate
        gate = torch.sigmoid(self.gate_fc(h_lid))

        fused_logits = gate * logits_cam + (1.0 - gate) * logits_lid
        return fused_logits, gate
