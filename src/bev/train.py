"""Training and evaluation routines for Camera-LiDAR BEV sensor fusion models."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.bev.data import generate_bev_dataset, split_bev_data, BEVDataset
from src.bev.corruptions import (
    apply_camera_fog,
    apply_camera_glare,
    apply_camera_occlusion,
    apply_lidar_rain,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TORCH_AVAILABLE = False

# -------------------------------------------------------------------
# Corruption dispatch
# -------------------------------------------------------------------

_CAMERA_CORRUPTIONS = {
    "fog": apply_camera_fog,
    "glare": apply_camera_glare,
    "occlusion": apply_camera_occlusion,
}

# Severity grid – 9 evenly spaced points in [0, 1]
S_GRID_POINTS = 9


def get_s_grid(n: int = S_GRID_POINTS) -> List[float]:
    """Return a list of n evenly-spaced severity levels in [0, 1]."""
    return np.linspace(0.0, 1.0, n).tolist()


def corrupt_cameras(cameras: np.ndarray, corruption: str, S: float, seed: int) -> np.ndarray:
    """Apply named camera corruption at severity S."""
    fn = _CAMERA_CORRUPTIONS[corruption]
    return fn(cameras, S=S, seed=seed)


# -------------------------------------------------------------------
# Train / eval helpers
# -------------------------------------------------------------------

def train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
) -> float:
    """Run one training epoch, returning mean cross-entropy loss."""
    model.train()
    total_loss = 0.0
    for cameras, lidars, labels, cam_removed in loader:
        cameras = cameras.to(device)
        lidars = lidars.to(device)
        labels = labels.to(device)
        cam_removed = cam_removed.to(device)

        optimizer.zero_grad()
        logits, _ = model(cameras, lidars, cam_removed=cam_removed)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def evaluate_bev_predictions(
    model: Any,
    test_cameras: np.ndarray,
    test_lidars: np.ndarray,
    test_labels: np.ndarray,
    device: Any,
    corruption: str,
    seed: int,
    grid_points: int = S_GRID_POINTS,
) -> Dict[str, Any]:
    """Evaluate BEV model: null baseline + 9-point camera-corruption severity grid.

    Parameters
    ----------
    model : trained BEVFixedFusionModel or BEVGatedFusionModel
    test_cameras / test_lidars / test_labels : raw numpy test arrays
    corruption : one of 'fog', 'glare', 'occlusion'
    seed : used for per-S corruption seed
    """
    model.eval()
    n_test = len(test_labels)
    num_classes = model.num_classes
    S_grid = get_s_grid(grid_points)

    # ---- Null: camera removed (LiDAR-only baseline B_null) ----
    with torch.no_grad():
        t_cam = torch.from_numpy(test_cameras).to(device)
        t_lid = torch.from_numpy(test_lidars).to(device)
        logits_null, _ = model(t_cam, t_lid, cam_removed=True)
        null_probs = torch.softmax(logits_null, dim=-1).cpu().numpy().tolist()

    # ---- 9-point severity grid ----
    probs_per_S: List[List[List[float]]] = [[] for _ in range(n_test)]
    for s_idx, S in enumerate(S_grid):
        corrupt_seed = seed * 1000 + s_idx
        corrupted_cams = corrupt_cameras(test_cameras, corruption, S=S, seed=corrupt_seed)
        with torch.no_grad():
            t_corr = torch.from_numpy(corrupted_cams).to(device)
            logits_s, _ = model(t_corr, t_lid, cam_removed=False)
            ps = torch.softmax(logits_s, dim=-1).cpu().numpy()
            for i in range(n_test):
                probs_per_S[i].append(ps[i].tolist())

    items = [
        {
            "item_id": i,
            "correct_choice_idx": int(test_labels[i]),
            "null_probs": {"removed": null_probs[i]},
            "probs_per_S": probs_per_S[i],
        }
        for i in range(n_test)
    ]

    return {"S_grid": S_grid, "items": items}


# -------------------------------------------------------------------
# Main experiment runner
# -------------------------------------------------------------------

def run_bev_experiment(
    model_name: str,
    corruption: str,
    dropout_p: float,
    seed: int,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_samples: int = 4000,
    n_train: int = 3000,
    n_val: int = 500,
    n_test: int = 500,
    hidden_dim: int = 64,
    num_classes: int = 4,
    smoke: bool = False,
    out_dir: Path = Path("results/raw/bev"),
    device_str: str = "cpu",
) -> Path:
    """Train and evaluate one BEV experiment cell; write-once predictions.

    Parameters
    ----------
    model_name : 'fixed' or 'gated'
    corruption  : 'fog', 'glare', or 'occlusion'
    dropout_p   : camera dropout probability during training (0.0 = no dropout)
    seed        : integer seed for dataset generation, training, and evaluation
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for BEV training.")

    from src.bev.model import BEVFixedFusionModel, BEVGatedFusionModel

    exp_name = f"{model_name}_{corruption}_p{dropout_p}_seed{seed}"
    exp_dir = out_dir / exp_name
    pred_file = exp_dir / "predictions.json"
    ckpt_file = exp_dir / "model.pt"

    if pred_file.exists():
        print(f"[SKIP] Predictions exist: {pred_file}")
        return pred_file

    exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    # ---- Data ----
    if smoke:
        n_samples = 400
        n_train, n_val, n_test = 240, 80, 80
        epochs = 1

    data = generate_bev_dataset(n_samples=n_samples, seed=seed)
    train_split, val_split, test_split = split_bev_data(
        data, n_train=n_train, n_val=n_val, n_test=n_test
    )

    train_ds = BEVDataset(
        train_split["cameras"],
        train_split["lidars"],
        train_split["labels"],
        camera_dropout_p=dropout_p,
        seed=seed,
    )
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # ---- Model ----
    torch.manual_seed(seed)
    if model_name == "fixed":
        model = BEVFixedFusionModel(hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    elif model_name == "gated":
        model = BEVGatedFusionModel(hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}. Use 'fixed' or 'gated'.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ---- Train ----
    print(f"Training BEV {exp_name} for {epochs} epoch(s) on {device}...")
    for ep in range(epochs):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        if (ep + 1) % 10 == 0 or smoke or epochs <= 5:
            print(f"  Epoch {ep+1}/{epochs} | Loss: {loss:.4f}")

    # ---- Save checkpoint ----
    torch.save(model.state_dict(), ckpt_file)

    # ---- Evaluate ----
    test_eval = evaluate_bev_predictions(
        model,
        test_cameras=test_split["cameras"],
        test_lidars=test_split["lidars"],
        test_labels=test_split["labels"],
        device=device,
        corruption=corruption,
        seed=seed,
        grid_points=S_GRID_POINTS,
    )

    pred_data = {
        "model": model_name,
        "corruption": corruption,
        "dropout_p": dropout_p,
        "seed": seed,
        "n_test": len(test_eval["items"]),
        "S_grid": test_eval["S_grid"],
        "items": test_eval["items"],
    }

    pred_file.write_text(json.dumps(pred_data, indent=2), encoding="utf-8")
    print(f"Saved predictions: {pred_file}")
    return pred_file
