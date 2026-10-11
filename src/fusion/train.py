"""Training and evaluation routines for late fusion models under modality dropout."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from src.fusion.data import (
    apply_modality_a_degradation,
    generate_synthetic_twoview_data,
    split_data,
    SyntheticTwoViewDataset,
    FashionMNISTTwoViewDataset,
)
from src.fusion.model import LateFusionModel
from src.io_utils import write_raw

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


def train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    for x_a, x_b, labels, a_removed in loader:
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        labels = labels.to(device)
        a_removed = a_removed.to(device)

        optimizer.zero_grad()
        logits, _ = model(x_a, x_b, a_removed=a_removed)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def evaluate_test_predictions(
    model: Any,
    test_dict: Dict[str, np.ndarray],
    device: Any,
    grid_points: int = 9,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate trained model on test set across 9-point degradation grid of modality A."""
    model.eval()
    x_a = test_dict["x_a"]
    x_b = test_dict["x_b"]
    labels = test_dict["labels"]
    n_test = len(labels)
    num_classes = model.num_classes

    S_grid = np.linspace(0.0, 1.0, grid_points).tolist()

    # 1. Null: modality A removed
    with torch.no_grad():
        t_xa = torch.from_numpy(x_a).to(device)
        t_xb = torch.from_numpy(x_b).to(device)
        logits_null, _ = model(t_xa, t_xb, a_removed=True)
        null_probs = torch.softmax(logits_null, dim=-1).cpu().numpy().tolist()

    # 2. Grid evaluation
    probs_per_S = [[] for _ in range(n_test)]
    for s_idx, S in enumerate(S_grid):
        degraded_xa = apply_modality_a_degradation(x_a, S=S, seed=seed * 1000 + int(S * 100))
        with torch.no_grad():
            t_deg_xa = torch.from_numpy(degraded_xa).to(device)
            logits_s, _ = model(t_deg_xa, t_xb, a_removed=False)
            ps = torch.softmax(logits_s, dim=-1).cpu().numpy()
            for i in range(n_test):
                probs_per_S[i].append(ps[i].tolist())

    items = []
    for i in range(n_test):
        items.append({
            "item_id": i,
            "correct_choice_idx": int(labels[i]),
            "null_probs": {
                "removed": null_probs[i],
            },
            "probs_per_S": probs_per_S[i],
        })

    return {
        "S_grid": S_grid,
        "items": items,
    }


def run_training_experiment(
    dataset_name: str,
    dropout_p: float,
    seed: int,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    smoke: bool = False,
    out_dir: Path = Path("results/raw/fusion"),
    device_str: str = "cpu",
) -> Path:
    """Train fusion model and write write-once checkpoint + predictions."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for running fusion training.")

    exp_dir = out_dir / f"{dataset_name}_p{dropout_p}_seed{seed}"
    pred_file = exp_dir / "predictions.json"
    ckpt_file = exp_dir / "model.pt"

    if pred_file.exists():
        print(f"Predictions already exist at {pred_file}. Skipping.")
        return pred_file

    exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    # 1. Prepare data
    if dataset_name == "synthetic":
        raw = generate_synthetic_twoview_data(
            n_samples=500 if smoke else 7000,
            dim_a=64,
            dim_b=64,
            n_classes=4,
            class_overlap=0.3,
            seed=seed,
        )
        if smoke:
            train_dict, val_dict, test_dict = split_data(raw, n_train=300, n_val=100, n_test=100)
        else:
            train_dict, val_dict, test_dict = split_data(raw, n_train=5000, n_val=1000, n_test=1000)

        dim_a, dim_b, n_classes = 64, 64, 4
        train_ds = SyntheticTwoViewDataset(train_dict, dropout_p=dropout_p, seed=seed)
    elif dataset_name == "fashionmnist":
        # VERIFY-ON-GPU: Fashion-MNIST dataset loading
        import torchvision  # type: ignore
        import torchvision.transforms as transforms  # type: ignore

        fmnist = torchvision.datasets.FashionMNIST(
            root="/kaggle/working/data" if Path("/kaggle/working").exists() else "./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        imgs = (fmnist.data.numpy() / 255.0).astype(np.float32)
        lbls = fmnist.targets.numpy().astype(np.int64)

        if smoke:
            imgs = imgs[:500]
            lbls = lbls[:500]
            n_train, n_val, n_test = 300, 100, 100
        else:
            n_train, n_val, n_test = 50000, 5000, 5000

        raw = {
            "x_a": imgs.reshape(len(imgs), 28, 28),
            "x_b": imgs.reshape(len(imgs), 28, 28),
            "labels": lbls,
        }
        train_dict, val_dict, test_dict = split_data(raw, n_train=n_train, n_val=n_val, n_test=n_test)
        dim_a, dim_b, n_classes = 392, 392, 10
        train_ds = FashionMNISTTwoViewDataset(train_dict["x_a"], train_dict["labels"], dropout_p=dropout_p, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # 2. Setup model
    torch.manual_seed(seed)
    model = LateFusionModel(dim_a=dim_a, dim_b=dim_b, hidden_dim=64, num_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 3. Train loop
    n_epochs = 1 if smoke else epochs
    print(f"Training {dataset_name} (p={dropout_p}, seed={seed}) for {n_epochs} epoch(s)...")
    for ep in range(n_epochs):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        if (ep + 1) % 5 == 0 or smoke:
            print(f"  Epoch {ep + 1}/{n_epochs} | Loss: {loss:.4f}")

    # 4. Save checkpoint (VERIFY-ON-GPU)
    torch.save(model.state_dict(), ckpt_file)

    # 5. Evaluate on test split (evaluated ONCE per protocol)
    test_eval = evaluate_test_predictions(model, test_dict, device=device, grid_points=9, seed=seed)
    pred_data = {
        "dataset": dataset_name,
        "dropout_p": dropout_p,
        "seed": seed,
        "n_test": len(test_eval["items"]),
        "S_grid": test_eval["S_grid"],
        "items": test_eval["items"],
    }
    write_raw(pred_file, pred_data)
    print(f"Saved test predictions to: {pred_file}")
    return pred_file
