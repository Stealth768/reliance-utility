import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Tuple, Union
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
    smooth,
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
    null_kinds: Optional[List[str]] = None,
    output_raw_dir: str = "results/raw/synthetic_v02",
    output_summary_csv: str = "results/derived/synthetic_summary_v02.csv",
) -> pd.DataFrame:
    """Execute synthetic validation benchmark across models, null kinds, and seeds.

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
    null_kinds : Optional[List[str]], default=None
        Null variants. Defaults to ['removed', 'zeros', 'mean'].
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

    if null_kinds is None:
        null_kinds = ["removed", "zeros", "mean"]

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

    summary_rows = []

    for seed in seeds:
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

    # Generate pooled summary
    df_pooled = generate_pooled_summary(raw_path, summary_path, S_grid=S_grid, n_boot=n_boot)
    print(f"Pooled summary written to {summary_path}")
    return df_pooled


def generate_pooled_summary(
    raw_dir: Union[str, Path],
    output_summary_csv: Union[str, Path],
    S_grid: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Aggregate raw evaluation results using pooled bootstrap across all seeds.

    Fixes the aggregation anomaly where point estimates lay outside averaged CIs
    by computing point estimates and bootstrap distributions on the pooled
    sample of all evaluations across seeds. Reports across-seed standard
    deviation in separate columns.

    Parameters
    ----------
    raw_dir : Union[str, Path]
        Directory containing raw JSON evaluation files.
    output_summary_csv : Union[str, Path]
        Destination CSV path.
    S_grid : Optional[np.ndarray], default=None
        Severity grid. Defaults to np.linspace(0, 1, 21).
    n_boot : int, default=1000
        Number of pooled bootstrap resamples.
    seed : int, default=42
        Random seed for the pooled bootstrap.

    Returns
    -------
    pd.DataFrame
        Pooled summary dataframe.
    """
    raw_path = Path(raw_dir)
    summary_path = Path(output_summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if S_grid is None:
        S_grid = np.linspace(0.0, 1.0, 21)

    # Discover all models and null kinds present in raw directory
    json_files = sorted(raw_path.glob("*.json"))
    records_by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for jf in json_files:
        if jf.name.startswith("manifest") or "run_" in jf.name:
            continue
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        group_key = (data["model_name"], data["null_kind"])
        if group_key not in records_by_group:
            records_by_group[group_key] = []
        records_by_group[group_key].append(data)

    pooled_rows = []

    # Sort groups: model order, then null_kind ('removed', 'zeros', 'mean')
    null_order = {"removed": 0, "zeros": 1, "mean": 2}
    model_order = {"FixedModel": 0, "PartialGatedModel": 1, "GatedModel": 2}

    sorted_keys = sorted(
        records_by_group.keys(),
        key=lambda k: (model_order.get(k[0], 99), null_order.get(k[1], 99)),
    )

    rng = np.random.default_rng(seed)

    for model_name, null_kind in sorted_keys:
        entries = records_by_group[(model_name, null_kind)]
        n_seeds = len(entries)

        per_seed_s = [e["S_star"] for e in entries]
        per_seed_mi = [e["MI"] for e in entries]
        per_seed_clean = [e["clean_acc"] for e in entries]
        per_seed_null = [e["acc_null"] for e in entries]

        # Concatenate per-item arrays across seeds
        acc_deg_list = [np.array(e["per_item_acc_degraded"]) for e in entries]  # (21, N)
        acc_null_list = [np.array(e["per_item_acc_null"]) for e in entries]      # (N,)
        jsd_list = [np.array(e["per_item_jsd"]) for e in entries]                # (21, N)

        acc_deg_pooled = np.concatenate(acc_deg_list, axis=1)  # (21, n_seeds * N)
        acc_null_pooled = np.concatenate(acc_null_list)        # (n_seeds * N,)
        jsd_pooled = np.concatenate(jsd_list, axis=1)          # (21, n_seeds * N)

        n_pooled = acc_deg_pooled.shape[1]

        # 1. Pooled point estimates
        mean_deg_pool = np.mean(acc_deg_pooled, axis=1)
        mean_null_pool = float(np.mean(acc_null_pooled))
        U_pool = utility(mean_deg_pool, mean_null_pool)
        S_star_pool = utility_crossing(U_pool, S_grid, window=3)

        R_pool = reliance(np.mean(jsd_pooled, axis=1))
        R_hat_pool = normalized_reliance(R_pool)
        MI_pool = misalignment_index(R_hat_pool, S_grid, S_star_pool)

        # 2. Check if U(S) approaches zero without clearly crossing
        U_smooth = smooth(U_pool, window=3)
        min_u_sm = float(np.min(U_smooth))
        approaches_zero = bool(min_u_sm > 0.0 and min_u_sm <= 0.05)

        # 3. Pooled bootstrap
        boot_s_samples: List[float] = []
        boot_mi_samples: List[float] = []
        none_count = 0

        for b in range(n_boot):
            b_idx = rng.integers(0, n_pooled, size=n_pooled)
            b_deg = np.mean(acc_deg_pooled[:, b_idx], axis=1)
            b_null = float(np.mean(acc_null_pooled[b_idx]))
            b_u = utility(b_deg, b_null)
            b_s = utility_crossing(b_u, S_grid, window=3)

            if b_s is None:
                none_count += 1
            else:
                boot_s_samples.append(float(b_s))
                b_r = np.mean(jsd_pooled[:, b_idx], axis=1)
                b_rh = normalized_reliance(b_r)
                b_mi = misalignment_index(b_rh, S_grid, b_s)
                if b_mi is not None:
                    boot_mi_samples.append(float(b_mi))

        no_crossing_fraction = none_count / n_boot

        # Bootstrap CIs and stability
        if len(boot_s_samples) == 0:
            s_lo, s_hi, s_boot_std = np.nan, np.nan, np.nan
        else:
            s_lo = float(np.percentile(boot_s_samples, 2.5))
            s_hi = float(np.percentile(boot_s_samples, 97.5))
            s_boot_std = float(np.std(boot_s_samples))

        if len(boot_mi_samples) == 0:
            mi_lo, mi_hi = np.nan, np.nan
        else:
            mi_lo = float(np.percentile(boot_mi_samples, 2.5))
            mi_hi = float(np.percentile(boot_mi_samples, 97.5))

        # Across-seed standard deviations
        valid_seed_s = [s for s in per_seed_s if s is not None]
        valid_seed_mi = [m for m in per_seed_mi if m is not None]
        s_seed_std = float(np.std(valid_seed_s)) if len(valid_seed_s) > 1 else np.nan
        mi_seed_std = float(np.std(valid_seed_mi)) if len(valid_seed_mi) > 1 else np.nan

        pooled_rows.append(
            {
                "model": model_name,
                "null_kind": null_kind,
                "S_star": S_star_pool if S_star_pool is not None else np.nan,
                "S_star_lo": s_lo,
                "S_star_hi": s_hi,
                "S_star_seed_std": s_seed_std,
                "S_star_boot_std": s_boot_std,
                "MI": MI_pool if MI_pool is not None else np.nan,
                "MI_lo": mi_lo,
                "MI_hi": mi_hi,
                "MI_seed_std": mi_seed_std,
                "no_crossing_fraction": no_crossing_fraction,
                "approaches_zero_without_crossing": approaches_zero,
                "clean_acc": float(np.mean(per_seed_clean)),
                "null_acc": float(np.mean(per_seed_null)),
            }
        )

    df_result = pd.DataFrame(pooled_rows)
    df_result.to_csv(summary_path, index=False)
    return df_result


if __name__ == "__main__":
    # Generate fixed v0.1 summary from existing results/raw/synthetic
    print("Generating synthetic_summary_v01_fixed.csv...")
    df_v01_fixed = generate_pooled_summary(
        raw_dir="results/raw/synthetic",
        output_summary_csv="results/derived/synthetic_summary_v01_fixed.csv",
    )
    print("v0.1 fixed summary generated successfully.")

    # Run v0.2 benchmark across 3 models x 3 nulls x 5 seeds
    print("\nRunning v0.2 benchmark into results/raw/synthetic_v02/...")
    df_v02 = run_experiment(
        null_kinds=["removed", "zeros", "mean"],
        output_raw_dir="results/raw/synthetic_v02",
        output_summary_csv="results/derived/synthetic_summary_v02.csv",
    )
    print("\nv0.2 benchmark complete.")
