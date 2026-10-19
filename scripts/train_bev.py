"""CLI runner for Camera-LiDAR BEV fusion experiments."""

import argparse
from pathlib import Path
import sys

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bev.train import run_bev_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Train Camera-LiDAR BEV fusion models")
    parser.add_argument(
        "--models",
        type=str,
        default="fixed,gated",
        help="Comma-separated models ('fixed', 'gated', default: 'fixed,gated')",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        default="fog,glare,occlusion",
        help="Comma-separated camera corruptions (default: 'fog,glare,occlusion')",
    )
    parser.add_argument(
        "--p",
        type=str,
        default="0.0,0.25",
        help="Comma-separated dropout probabilities p (default: '0.0,0.25')",
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
        default=25,
        help="Number of epochs (default: 25)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/raw/bev",
        help="Output directory for raw checkpoints and predictions",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test (1 epoch, tiny sample)",
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
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    corruptions = [x.strip() for x in args.corruptions.split(",") if x.strip()]
    p_values = [float(x.strip()) for x in args.p.split(",") if x.strip()]
    seed_values = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out)

    print("==================================================")
    print("BEV CAMERA-LIDAR FUSION EXPERIMENTS")
    print(f"Models: {models}")
    print(f"Corruptions: {corruptions}")
    print(f"p grid: {p_values}")
    print(f"Seeds: {seed_values}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {out_dir}")
    print(f"Device: {args.device}")
    print(f"Smoke mode: {args.smoke}")
    print("==================================================")

    total_runs = len(models) * len(corruptions) * len(p_values) * len(seed_values)
    run_idx = 0

    for m in models:
        for corr in corruptions:
            for p in p_values:
                for seed in seed_values:
                    run_idx += 1
                    print(f"\n>>> [{run_idx}/{total_runs}] Model={m}, Corr={corr}, p={p}, Seed={seed} <<<")
                    run_bev_experiment(
                        model_name=m,
                        corruption=corr,
                        dropout_p=p,
                        seed=seed,
                        epochs=args.epochs,
                        smoke=args.smoke,
                        out_dir=out_dir,
                        device_str=args.device,
                    )


if __name__ == "__main__":
    main()
