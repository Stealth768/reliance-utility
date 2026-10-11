"""CLI runner for modality-dropout fusion experiments."""

import argparse
from pathlib import Path
import sys

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fusion.train import run_training_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Train late fusion model with modality dropout")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "fashionmnist"],
        help="Dataset name ('synthetic' on CPU or 'fashionmnist' on GPU)",
    )
    parser.add_argument(
        "--p",
        type=str,
        default="0,0.1,0.25,0.5",
        help="Comma-separated dropout probabilities p (default: '0,0.1,0.25,0.5')",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds (default: '0,1,2,3,4')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs (default: 30 for synthetic, 20 for fashionmnist)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/raw/fusion",
        help="Output directory for raw checkpoints and predictions",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test (1 epoch, 500 samples)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or 'cuda')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    p_values = [float(x.strip()) for x in args.p.split(",") if x.strip()]
    seed_values = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out)

    print("==================================================")
    print(f"MODALITY-DROPOUT FUSION EXPERIMENTS")
    print(f"Dataset: {args.dataset}")
    print(f"p grid: {p_values}")
    print(f"Seeds: {seed_values}")
    print(f"Output: {out_dir}")
    print(f"Smoke mode: {args.smoke}")
    print("==================================================")

    for p in p_values:
        for seed in seed_values:
            print(f"\n>>> Running p={p}, seed={seed} <<<")
            epochs = 20 if args.dataset == "fashionmnist" else args.epochs
            run_training_experiment(
                dataset_name=args.dataset,
                dropout_p=p,
                seed=seed,
                epochs=epochs,
                smoke=args.smoke,
                out_dir=out_dir,
                device_str=args.device,
            )


if __name__ == "__main__":
    main()
