import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
from src.synthetic.models_v03 import OracleGatedModel, SharpGatedModel


def get_default_v03_models() -> List[Tuple[str, Any]]:
    """Return the set of models evaluated in synthetic benchmark iteration v0.3."""
    return [
        ("FixedModel", FixedModel),
        ("GatedModel", GatedModel),
        ("PartialGatedModel", PartialGatedModel),
        ("OracleGatedModel", OracleGatedModel),
        ("SharpGatedModel_0.5", lambda: SharpGatedModel(0.5)),
        ("SharpGatedModel_1.0", lambda: SharpGatedModel(1.0)),
        ("SharpGatedModel_2.0", lambda: SharpGatedModel(2.0)),
        ("SharpGatedModel_4.0", lambda: SharpGatedModel(4.0)),
        ("SharpGatedModel_8.0", lambda: SharpGatedModel(8.0)),
        ("SharpGatedModel_16.0", lambda: SharpGatedModel(16.0)),
    ]


def run_experiment(
    n_train: int = 2000,
    n_val: int = 1000,
    n_test: int = 2000,
    n_boot: int = 1000,
    seeds: Optional[List[int]] = None,
    null_kinds: Optional[List[str]] = None,
    models: Optional[List[Tuple[str, Any]]] = None,
    output_raw_dir: str = "results/raw/synthetic_v03",
    output_summary_csv: str = "results/derived/synthetic_summary_v03.csv",
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
    models : Optional[List[Tuple[str, Any]]], default=None
        List of (model_name, model_class_or_factory) pairs.
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

    if models is None:
        models = get_default_v03_models()

    raw_path = Path(output_raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    summary_path = Path(output_summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    S_grid = np.linspace(0.0, 1.0, 21)

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
                if hasattr(model, "set_severity"):
                    model.set_severity(None)

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
                    if hasattr(model, "set_severity"):
                        model.set_severity(S)

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

                if hasattr(model, "set_severity"):
                    model.set_severity(None)

                # 3. Full-sample point estimates
                acc_deg = [float(np.mean(acc_matrix[s])) for s in range(len(S_grid))]
                U = utility(acc_deg, acc_null)
                D = [float(np.mean(jsd_matrix[s])) for s in range(len(S_grid))]
                R = reliance(D)
                R_hat = normalized_reliance(R)
                S_star = utility_crossing(U, S_grid, window=3)
                MI = misalignment_index(R_hat, S_grid, S_star)

                # 4. Bootstrap CI over test items (single seed)
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

    # Generate pooled and hierarchical summary
    df_pooled = generate_pooled_summary(raw_path, summary_path, S_grid=S_grid, n_boot=n_boot)
    print(f"Summary written to {summary_path}")
    return df_pooled


def generate_pooled_summary(
    raw_dir: Union[str, Path],
    output_summary_csv: Union[str, Path],
    S_grid: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Aggregate raw evaluation results using pooled and hierarchical bootstrap across all seeds.

    Computes:
    - Pooled point estimates
    - Pooled bootstrap confidence intervals
    - Hierarchical bootstrap confidence intervals (resample seeds, then items within chosen seeds)
    - Per-seed point estimate statistics (min, max, std)
    - Bootstrap stability (std of S* across resamples)
    - Non-crossing fraction

    Parameters
    ----------
    raw_dir : Union[str, Path]
        Directory containing raw JSON evaluation files.
    output_summary_csv : Union[str, Path]
        Destination CSV path.
    S_grid : Optional[np.ndarray], default=None
        Severity grid. Defaults to np.linspace(0, 1, 21).
    n_boot : int, default=1000
        Number of bootstrap resamples.
    seed : int, default=42
        Random seed for the bootstrap.

    Returns
    -------
    pd.DataFrame
        Summary dataframe.
    """
    raw_path = Path(raw_dir)
    summary_path = Path(output_summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if S_grid is None:
        S_grid = np.linspace(0.0, 1.0, 21)

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

    null_order = {"removed": 0, "zeros": 1, "mean": 2}
    model_order = {
        "FixedModel": 0,
        "PartialGatedModel": 1,
        "GatedModel": 2,
        "OracleGatedModel": 3,
        "SharpGatedModel_0.5": 4,
        "SharpGatedModel_1.0": 5,
        "SharpGatedModel_2.0": 6,
        "SharpGatedModel_4.0": 7,
        "SharpGatedModel_8.0": 8,
        "SharpGatedModel_16.0": 9,
    }

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

        # Per-seed arrays
        acc_deg_by_seed = [np.array(e["per_item_acc_degraded"]) for e in entries]  # list of (21, N)
        acc_null_by_seed = [np.array(e["per_item_acc_null"]) for e in entries]    # list of (N,)
        jsd_by_seed = [np.array(e["per_item_jsd"]) for e in entries]              # list of (21, N)

        n_items_per_seed = acc_null_by_seed[0].shape[0]

        # Pooled arrays
        acc_deg_pooled = np.concatenate(acc_deg_by_seed, axis=1)  # (21, n_seeds * N)
        acc_null_pooled = np.concatenate(acc_null_by_seed)        # (n_seeds * N,)
        jsd_pooled = np.concatenate(jsd_by_seed, axis=1)          # (21, n_seeds * N)
        n_pooled = acc_deg_pooled.shape[1]

        # 1. Pooled point estimates
        mean_deg_pool = np.mean(acc_deg_pooled, axis=1)
        mean_null_pool = float(np.mean(acc_null_pooled))
        U_pool = utility(mean_deg_pool, mean_null_pool)
        S_star_pool = utility_crossing(U_pool, S_grid, window=3)

        R_pool = reliance(np.mean(jsd_pooled, axis=1))
        R_hat_pool = normalized_reliance(R_pool)
        MI_pool = misalignment_index(R_hat_pool, S_grid, S_star_pool)

        # 2. Check if U(S) approaches zero without crossing
        U_smooth = smooth(U_pool, window=3)
        min_u_sm = float(np.min(U_smooth))
        approaches_zero = bool(min_u_sm > 0.0 and min_u_sm <= 0.05)

        # 3. Pooled bootstrap
        boot_s_samples: List[float] = []
        boot_mi_samples: List[float] = []
        pooled_none_count = 0

        for b in range(n_boot):
            b_idx = rng.integers(0, n_pooled, size=n_pooled)
            b_deg = np.mean(acc_deg_pooled[:, b_idx], axis=1)
            b_null = float(np.mean(acc_null_pooled[b_idx]))
            b_u = utility(b_deg, b_null)
            b_s = utility_crossing(b_u, S_grid, window=3)

            if b_s is None:
                pooled_none_count += 1
            else:
                boot_s_samples.append(float(b_s))
                b_r = np.mean(jsd_pooled[:, b_idx], axis=1)
                b_rh = normalized_reliance(b_r)
                b_mi = misalignment_index(b_rh, S_grid, b_s)
                if b_mi is not None:
                    boot_mi_samples.append(float(b_mi))

        pooled_no_crossing_frac = pooled_none_count / n_boot

        # 4. Hierarchical bootstrap (resample seeds, then items within chosen seeds)
        hier_s_samples: List[float] = []
        hier_mi_samples: List[float] = []
        hier_none_count = 0

        for b in range(n_boot):
            chosen_seeds = rng.integers(0, n_seeds, size=n_seeds)
            deg_means = []
            null_means = []
            jsd_means = []

            for s_idx in chosen_seeds:
                item_idx = rng.integers(0, n_items_per_seed, size=n_items_per_seed)
                deg_means.append(np.mean(acc_deg_by_seed[s_idx][:, item_idx], axis=1))
                null_means.append(np.mean(acc_null_by_seed[s_idx][item_idx]))
                jsd_means.append(np.mean(jsd_by_seed[s_idx][:, item_idx], axis=1))

            b_h_deg = np.mean(deg_means, axis=0)
            b_h_null = float(np.mean(null_means))
            b_h_jsd = np.mean(jsd_means, axis=0)

            b_h_u = utility(b_h_deg, b_h_null)
            b_h_s = utility_crossing(b_h_u, S_grid, window=3)

            if b_h_s is None:
                hier_none_count += 1
            else:
                hier_s_samples.append(float(b_h_s))
                b_h_r = reliance(b_h_jsd)
                b_h_rh = normalized_reliance(b_h_r)
                b_h_mi = misalignment_index(b_h_rh, S_grid, b_h_s)
                if b_h_mi is not None:
                    hier_mi_samples.append(float(b_h_mi))

        hier_no_crossing_frac = hier_none_count / n_boot

        # Bounds computation
        # Pooled CIs
        s_pool_lo = float(np.percentile(boot_s_samples, 2.5)) if boot_s_samples else np.nan
        s_pool_hi = float(np.percentile(boot_s_samples, 97.5)) if boot_s_samples else np.nan
        s_boot_std = float(np.std(boot_s_samples)) if boot_s_samples else np.nan

        mi_pool_lo = float(np.percentile(boot_mi_samples, 2.5)) if boot_mi_samples else np.nan
        mi_pool_hi = float(np.percentile(boot_mi_samples, 97.5)) if boot_mi_samples else np.nan

        # Hierarchical CIs
        s_hier_lo = float(np.percentile(hier_s_samples, 2.5)) if hier_s_samples else np.nan
        s_hier_hi = float(np.percentile(hier_s_samples, 97.5)) if hier_s_samples else np.nan

        mi_hier_lo = float(np.percentile(hier_mi_samples, 2.5)) if hier_mi_samples else np.nan
        mi_hier_hi = float(np.percentile(hier_mi_samples, 97.5)) if hier_mi_samples else np.nan

        # Per-seed stats
        valid_seed_s = [s for s in per_seed_s if s is not None]
        valid_seed_mi = [m for m in per_seed_mi if m is not None]

        s_seed_min = float(np.min(valid_seed_s)) if valid_seed_s else np.nan
        s_seed_max = float(np.max(valid_seed_s)) if valid_seed_s else np.nan
        s_seed_std = float(np.std(valid_seed_s)) if len(valid_seed_s) > 1 else np.nan

        mi_seed_min = float(np.min(valid_seed_mi)) if valid_seed_mi else np.nan
        mi_seed_max = float(np.max(valid_seed_mi)) if valid_seed_mi else np.nan
        mi_seed_std = float(np.std(valid_seed_mi)) if len(valid_seed_mi) > 1 else np.nan

        # Formatted string representations for reporting
        def fmt_ci(lo: float, hi: float) -> str:
            if np.isnan(lo) or np.isnan(hi):
                return "no crossing"
            return f"[{lo:.2f}, {hi:.2f}]"

        def fmt_min_max(mn: float, mx: float) -> str:
            if np.isnan(mn) or np.isnan(mx):
                return "no crossing"
            return f"({mn:.2f}, {mx:.2f})"

        pooled_rows.append(
            {
                "model": model_name,
                "null_kind": null_kind,
                # S* metrics
                "S_star": S_star_pool if S_star_pool is not None else np.nan,
                "S_star_lo": s_pool_lo,
                "S_star_hi": s_pool_hi,
                "S_star_pooled_CI": fmt_ci(s_pool_lo, s_pool_hi),
                "S_star_hier_lo": s_hier_lo,
                "S_star_hier_hi": s_hier_hi,
                "S_star_hier_CI": fmt_ci(s_hier_lo, s_hier_hi),
                "S_star_seed_min": s_seed_min,
                "S_star_seed_max": s_seed_max,
                "S_star_seed_min_max": fmt_min_max(s_seed_min, s_seed_max),
                "S_star_seed_std": s_seed_std,
                "S_star_boot_std": s_boot_std,
                # MI metrics
                "MI": MI_pool if MI_pool is not None else np.nan,
                "MI_lo": mi_pool_lo,
                "MI_hi": mi_pool_hi,
                "MI_pooled_CI": fmt_ci(mi_pool_lo, mi_pool_hi),
                "MI_hier_lo": mi_hier_lo,
                "MI_hier_hi": mi_hier_hi,
                "MI_hier_CI": fmt_ci(mi_hier_lo, mi_hier_hi),
                "MI_seed_min": mi_seed_min,
                "MI_seed_max": mi_seed_max,
                "MI_seed_min_max": fmt_min_max(mi_seed_min, mi_seed_max),
                "MI_seed_std": mi_seed_std,
                # Quality & diagnostic metrics
                "no_crossing_fraction": pooled_no_crossing_frac,
                "no_crossing_fraction_hier": hier_no_crossing_frac,
                "approaches_zero_without_crossing": approaches_zero,
                "clean_acc": float(np.mean(per_seed_clean)),
                "null_acc": float(np.mean(per_seed_null)),
            }
        )

    df_result = pd.DataFrame(pooled_rows)
    df_result.to_csv(summary_path, index=False)
    return df_result


if __name__ == "__main__":
    df_sum = run_experiment(
        output_raw_dir="results/raw/synthetic_v03",
        output_summary_csv="results/derived/synthetic_summary_v03.csv",
    )
    print(df_sum)
