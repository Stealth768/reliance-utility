import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from scripts.make_manifest import write_manifest
from src.io_utils import write_raw
from src.metrics import (
    bootstrap_ci,
    jsd,
    misalignment_index,
    normalized_reliance,
    reliance,
    utility,
    utility_crossing,
)
from src.synthetic.data import degrade_A, make_data, null_A
from src.synthetic.models import FixedModel, GatedModel, PartialGatedModel


def run_experiment(
    n_train: int = 2000,
    n_val: int = 1000,
    n_test: int = 2000,
    n_boot: int = 1000,
    seeds: Optional[List[int]] = None,
    output_raw_dir: str = "results/raw/synthetic",
    output_summary_csv: str = "results/derived/synthetic_summary.csv",
) -> pd.DataFrame:
    """Execute the full synthetic validation benchmark across models, null kinds, and seeds.

    Parameters
    ----------
    n_train : int, default=2000
        Number of clean training samples.
    n_val : int, default=1000
        Number of validation samples for gate calibration.
    n_test : int, default=2000
        Number of test samples (bootstrap units).
    n_boot : int, default=1000
        Number of bootstrap iterations for confidence intervals.
    seeds : Optional[List[int]], default=None
        List of data seeds. Defaults to [0, 1, 2, 3, 4].
    output_raw_dir : str
        Directory to store immutable raw output files.
    output_summary_csv : str
        Path to output derived summary CSV.

    Returns
    -------
    pd.DataFrame
        Derived summary dataframe.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    raw_path = Path(output_raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    summary_path = Path(output_summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    S_grid = np.linspace(0.0, 1.0, 21)
    models = [
        ("FixedModel", FixedModel),
        ("GatedModel", GatedModel),
        ("PartialGatedModel", PartialGatedModel),
    ]
    null_kinds = ["zeros", "mean"]

    summary_rows = []

    for seed in seeds:
        # Generate datasets for this seed
        train = make_data(n=n_train, k=4, d=8, seed=seed)
        val = make_data(n=n_val, k=4, d=8, seed=100 + seed, mu_A=train.mu_A, mu_B=train.mu_B)
        test = make_data(n=n_test, k=4, d=8, seed=200 + seed, mu_A=train.mu_A, mu_B=train.mu_B)
        tau = float(np.std(train.X_A))

        y_test = test.y
        n_items = len(y_test)

        for model_name, ModelClass in models:
            model = ModelClass().fit(train.X_A, train.X_B, train.y)
            if hasattr(model, "calibrate_gate"):
                model.calibrate_gate(val.X_A, seed=300 + seed)

            for null_kind in null_kinds:
                raw_filename = f"{model_name}_{null_kind}_seed{seed}.json"
                raw_file = raw_path / raw_filename

                # 1. Null baseline predictions
                X_null = null_A(null_kind, test.X_A, dataset_mean=np.mean(train.X_A, axis=0))
                p_null = model.predict_proba(X_null, test.X_B)
                acc_null_per_item = (np.argmax(p_null, axis=1) == y_test).astype(int)
                acc_null = float(np.mean(acc_null_per_item))

                # 2. Predictions across severity grid S
                acc_matrix = np.zeros((len(S_grid), n_items), dtype=int)
                jsd_matrix = np.zeros((len(S_grid), n_items), dtype=float)

                for step_idx, S in enumerate(S_grid):
                    Xs = degrade_A(
                        test.X_A,
                        S=S,
                        seed=1000 * seed + int(round(S * 100)),
                        tau=tau,
                    )
                    ps = model.predict_proba(Xs, test.X_B)
                    acc_matrix[step_idx] = (np.argmax(ps, axis=1) == y_test).astype(int)

                    # Per-item JSD between ps[i] and p_null[i]
                    m = 0.5 * (ps + p_null)
                    kl_p = np.sum(
                        ps * np.log2(np.maximum(ps, 1e-12) / np.maximum(m, 1e-12)),
                        axis=-1,
                    )
                    kl_null = np.sum(
                        p_null * np.log2(np.maximum(p_null, 1e-12) / np.maximum(m, 1e-12)),
                        axis=-1,
                    )
                    jsd_matrix[step_idx] = np.clip(0.5 * (kl_p + kl_null), 0.0, 1.0)

                # 3. Full-sample point estimates
                acc_deg = [float(np.mean(acc_matrix[s])) for s in range(len(S_grid))]
                U = utility(acc_deg, acc_null)
                D = [float(np.mean(jsd_matrix[s])) for s in range(len(S_grid))]
                R = reliance(D)
                R_hat = normalized_reliance(R)
                S_star = utility_crossing(U, S_grid, window=3)
                MI = misalignment_index(R_hat, S_grid, S_star)

                # 4. Bootstrap CI over test items
                def eval_resample(indices: np.ndarray):
                    acc_d_b = np.mean(acc_matrix[:, indices], axis=1)
                    acc_n_b = float(np.mean(acc_null_per_item[indices]))
                    u_b = utility(acc_d_b, acc_n_b)
                    s_b = utility_crossing(u_b, S_grid, window=3)
                    if s_b is None:
                        return None, None
                    r_b = np.mean(jsd_matrix[:, indices], axis=1)
                    r_hat_b = normalized_reliance(r_b)
                    mi_b = misalignment_index(r_hat_b, S_grid, s_b)
                    return s_b, mi_b

                boot_indices = np.arange(n_items)
                s_star_ci = bootstrap_ci(
                    lambda idx: eval_resample(idx)[0],
                    boot_indices,
                    n_boot=n_boot,
                    seed=42 + seed,
                )
                mi_ci = bootstrap_ci(
                    lambda idx: eval_resample(idx)[1],
                    boot_indices,
                    n_boot=n_boot,
                    seed=42 + seed,
                )

                # 5. Write raw per-item outputs ONCE
                raw_data: Dict[str, Any] = {
                    "model_name": model_name,
                    "null_kind": null_kind,
                    "data_seed": seed,
                    "n_test": n_test,
                    "n_boot": n_boot,
                    "S_grid": S_grid.tolist(),
                    "acc_degraded": acc_deg,
                    "acc_null": acc_null,
                    "U": U.tolist(),
                    "R": R.tolist(),
                    "R_hat": R_hat.tolist(),
                    "S_star": S_star,
                    "S_star_lo": s_star_ci.lo,
                    "S_star_hi": s_star_ci.hi,
                    "MI": MI,
                    "MI_lo": mi_ci.lo,
                    "MI_hi": mi_ci.hi,
                    "no_crossing_fraction": s_star_ci.no_crossing_fraction,
                    "clean_acc": acc_deg[0],
                    "per_item_acc_degraded": acc_matrix.tolist(),
                    "per_item_acc_null": acc_null_per_item.tolist(),
                    "per_item_jsd": jsd_matrix.tolist(),
                }

                if not raw_file.exists():
                    write_raw(raw_file, raw_data)

                # Write run manifest
                config_dict = {
                    "model_name": model_name,
                    "null_kind": null_kind,
                    "data_seed": seed,
                    "n_train": n_train,
                    "n_val": n_val,
                    "n_test": n_test,
                    "n_boot": n_boot,
                    "severity_points": len(S_grid),
                }
                manifest_dir = raw_path / f"run_{model_name}_{null_kind}_seed{seed}"
                write_manifest(manifest_dir, config_dict, seed=seed, model_name=model_name)

                # Record summary row
                summary_rows.append(
                    {
                        "model": model_name,
                        "null_kind": null_kind,
                        "seed": seed,
                        "clean_acc": acc_deg[0],
                        "null_acc": acc_null,
                        "S_star": S_star,
                        "S_star_lo": s_star_ci.lo,
                        "S_star_hi": s_star_ci.hi,
                        "MI": MI,
                        "MI_lo": mi_ci.lo,
                        "MI_hi": mi_ci.hi,
                        "no_crossing_fraction": s_star_ci.no_crossing_fraction,
                    }
                )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary written to {summary_path}")
    return df_summary


if __name__ == "__main__":
    df = run_experiment()
    print("\n=== SYNTHETIC VALIDATION SUMMARY ===")
    print(df.to_string(index=False))
