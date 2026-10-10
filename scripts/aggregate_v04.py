"""Re-aggregate v0.3 raw data with unclipped MI -> synthetic_summary_v04.csv.

Also runs:
  - Task 3: paired bootstrap analysis for null='removed'
  - Task 4: success-criterion evaluation and figure generation
  - Task 5 updates logged separately in DEVIATIONS.md

Run from repo root:
    python scripts/aggregate_v04.py
"""

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import (
    misalignment_index,
    normalized_reliance,
    reliance,
    smooth,
    utility,
    utility_crossing,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

S_GRID = np.linspace(0.0, 1.0, 21)
N_BOOT = 1000
RNG_SEED = 42

NULL_ORDER  = {"removed": 0, "zeros": 1, "mean": 2}
MODEL_ORDER = {
    "FixedModel": 0, "PartialGatedModel": 1, "GatedModel": 2,
    "OracleGatedModel": 3,
    "SharpGatedModel_0.5": 4, "SharpGatedModel_1.0": 5,
    "SharpGatedModel_2.0": 6, "SharpGatedModel_4.0": 7,
    "SharpGatedModel_8.0": 8, "SharpGatedModel_16.0": 9,
}


def fmt_ci(lo, hi):
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)):
        return "no crossing"
    return f"[{lo:.3f}, {hi:.3f}]"


def fmt_val(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "no crossing"
    return f"{v:.4f}"


def _eval_pool(acc_deg_pool, acc_null_pool, jsd_pool, S_grid):
    """Compute point estimates from pooled arrays."""
    U = utility(np.mean(acc_deg_pool, axis=1), float(np.mean(acc_null_pool)))
    s_star = utility_crossing(U, S_grid, window=3)
    R = reliance(np.mean(jsd_pool, axis=1))
    R_hat = normalized_reliance(R)
    mi = misalignment_index(R_hat, S_grid, s_star)
    return s_star, mi, R_hat, U


def _pooled_boot(acc_deg_pool, acc_null_pool, jsd_pool, S_grid, n_boot, rng):
    boot_s, boot_mi = [], []
    n = acc_deg_pool.shape[1]
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        U_b = utility(np.mean(acc_deg_pool[:, idx], axis=1),
                      float(np.mean(acc_null_pool[idx])))
        s_b = utility_crossing(U_b, S_grid, window=3)
        if s_b is None:
            continue
        R_b = reliance(np.mean(jsd_pool[:, idx], axis=1))
        rh_b = normalized_reliance(R_b)
        mi_b = misalignment_index(rh_b, S_grid, s_b)
        if mi_b is not None:
            boot_s.append(s_b)
            boot_mi.append(mi_b)
    return boot_s, boot_mi


def _hier_boot(acc_deg_by_seed, acc_null_by_seed, jsd_by_seed, S_grid, n_boot, rng):
    n_seeds = len(acc_deg_by_seed)
    n_items = acc_null_by_seed[0].shape[0]
    boot_s, boot_mi = [], []
    for _ in range(n_boot):
        chosen = rng.integers(0, n_seeds, size=n_seeds)
        dlist, nlist, jlist = [], [], []
        for s in chosen:
            idx = rng.integers(0, n_items, size=n_items)
            dlist.append(np.mean(acc_deg_by_seed[s][:, idx], axis=1))
            nlist.append(np.mean(acc_null_by_seed[s][idx]))
            jlist.append(np.mean(jsd_by_seed[s][:, idx], axis=1))
        U_b = utility(np.mean(dlist, axis=0), float(np.mean(nlist)))
        s_b = utility_crossing(U_b, S_grid, window=3)
        if s_b is None:
            continue
        R_b = reliance(np.mean(jlist, axis=0))
        rh_b = normalized_reliance(R_b)
        mi_b = misalignment_index(rh_b, S_grid, s_b)
        if mi_b is not None:
            boot_s.append(s_b)
            boot_mi.append(mi_b)
    return boot_s, boot_mi


def pct_ci(samples, lo=2.5, hi=97.5):
    if not samples:
        return None, None
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


# ──────────────────────────────────────────────────────────────────────────────
# Load raw v0.3 data
# ──────────────────────────────────────────────────────────────────────────────

RAW_DIR = REPO_ROOT / "results" / "raw" / "synthetic_v03"
OUT_CSV  = REPO_ROOT / "results" / "derived" / "synthetic_summary_v04.csv"
FIG_DIR  = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

json_files = sorted(RAW_DIR.glob("*.json"))
records_by_group: dict = {}
for jf in json_files:
    if "manifest" in jf.name or "run_" in jf.name:
        continue
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)
    key = (data["model_name"], data["null_kind"])
    records_by_group.setdefault(key, []).append(data)

sorted_keys = sorted(
    records_by_group.keys(),
    key=lambda k: (MODEL_ORDER.get(k[0], 99), NULL_ORDER.get(k[1], 99)),
)

# ──────────────────────────────────────────────────────────────────────────────
# Build summary rows
# ──────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(RNG_SEED)
rows = []

for model_name, null_kind in sorted_keys:
    entries = records_by_group[(model_name, null_kind)]

    acc_deg_by_seed = [np.array(e["per_item_acc_degraded"]) for e in entries]
    acc_null_by_seed = [np.array(e["per_item_acc_null"])    for e in entries]
    jsd_by_seed      = [np.array(e["per_item_jsd"])         for e in entries]
    n_seeds = len(entries)

    acc_deg_pool  = np.concatenate(acc_deg_by_seed,  axis=1)
    acc_null_pool = np.concatenate(acc_null_by_seed)
    jsd_pool      = np.concatenate(jsd_by_seed,      axis=1)

    # Point estimates
    s_star, mi, R_hat_pool, U_pool = _eval_pool(
        acc_deg_pool, acc_null_pool, jsd_pool, S_GRID)

    # U approaches 0 without crossing?
    U_sm = smooth(U_pool, window=3)
    approaches_zero = bool(np.min(U_sm) > 0.0 and np.min(U_sm) <= 0.05)

    # Per-seed point estimates
    per_seed_s  = [e["S_star"] for e in entries]
    per_seed_mi_raw = []
    for e in entries:
        rh = np.array(e["R_hat"])
        sg = np.array(e["S_grid"])
        ss = e["S_star"]
        per_seed_mi_raw.append(misalignment_index(rh, sg, ss))

    valid_s  = [v for v in per_seed_s  if v is not None]
    valid_mi = [v for v in per_seed_mi_raw if v is not None]

    # Pooled bootstrap
    pb_s, pb_mi = _pooled_boot(acc_deg_pool, acc_null_pool, jsd_pool,
                               S_GRID, N_BOOT, rng)
    # Hierarchical bootstrap
    hb_s, hb_mi = _hier_boot(acc_deg_by_seed, acc_null_by_seed, jsd_by_seed,
                              S_GRID, N_BOOT, rng)

    s_pool_lo,  s_pool_hi  = pct_ci(pb_s)
    mi_pool_lo, mi_pool_hi = pct_ci(pb_mi)
    s_hier_lo,  s_hier_hi  = pct_ci(hb_s)
    mi_hier_lo, mi_hier_hi = pct_ci(hb_mi)

    rows.append({
        "model":    model_name,
        "null_kind": null_kind,
        "n_seeds":  n_seeds,
        # S*
        "S_star":         s_star if s_star is not None else float("nan"),
        "S_star_lo":      s_pool_lo,
        "S_star_hi":      s_pool_hi,
        "S_star_pooled_CI": fmt_ci(s_pool_lo, s_pool_hi),
        "S_star_hier_lo": s_hier_lo,
        "S_star_hier_hi": s_hier_hi,
        "S_star_hier_CI": fmt_ci(s_hier_lo, s_hier_hi),
        "S_star_seed_min": min(valid_s) if valid_s else float("nan"),
        "S_star_seed_max": max(valid_s) if valid_s else float("nan"),
        "S_star_seed_std": float(np.std(valid_s)) if len(valid_s) > 1 else float("nan"),
        "S_star_boot_std": float(np.std(pb_s))   if pb_s else float("nan"),
        # MI (unclipped)
        "MI":         mi if mi is not None else float("nan"),
        "MI_lo":      mi_pool_lo,
        "MI_hi":      mi_pool_hi,
        "MI_pooled_CI": fmt_ci(mi_pool_lo, mi_pool_hi),
        "MI_hier_lo": mi_hier_lo,
        "MI_hier_hi": mi_hier_hi,
        "MI_hier_CI": fmt_ci(mi_hier_lo, mi_hier_hi),
        "MI_seed_min": min(valid_mi) if valid_mi else float("nan"),
        "MI_seed_max": max(valid_mi) if valid_mi else float("nan"),
        "MI_seed_std": float(np.std(valid_mi)) if len(valid_mi) > 1 else float("nan"),
        # diagnostics
        "no_crossing_fraction_pooled": len(pb_s) / N_BOOT if pb_s else float("nan"),
        "no_crossing_fraction_hier":   len(hb_s) / N_BOOT if hb_s else float("nan"),
        "approaches_zero_without_crossing": approaches_zero,
        "clean_acc": float(np.mean([e["clean_acc"] for e in entries])),
        "null_acc":  float(np.mean([e["acc_null"]  for e in entries])),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}")

# ──────────────────────────────────────────────────────────────────────────────
# Print Task 2 table (null=removed)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("TASK 2 -- Results table for null='removed'")
print("=" * 100)
rem = df[df["null_kind"] == "removed"][
    ["model", "S_star", "S_star_pooled_CI", "S_star_hier_CI",
     "MI", "MI_pooled_CI", "MI_hier_CI", "MI_seed_std", "no_crossing_fraction_pooled"]
].copy()
rem["MI"] = rem["MI"].map(lambda v: f"{v:.4f}" if not math.isnan(v) else "no crossing")
print(rem.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# Task 3 -- Paired bootstrap analysis for null='removed'
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("TASK 3 -- Paired bootstrap: MI differences (null='removed')")
print("=" * 100)

def get_per_seed_mi(model_name, null_kind="removed"):
    entries = records_by_group.get((model_name, null_kind), [])
    results = []
    for e in sorted(entries, key=lambda x: x["data_seed"]):
        rh = np.array(e["R_hat"])
        sg = np.array(e["S_grid"])
        ss = e["S_star"]
        mi = misalignment_index(rh, sg, ss)
        results.append((e["data_seed"], mi))
    return results

def paired_boot_diff(entries_a, entries_b, n_boot=N_BOOT, seed=RNG_SEED):
    """Bootstrap the mean MI(a) - MI(b) pairing per seed then resampling items."""
    # Sort by seed
    entries_a = sorted(entries_a, key=lambda e: e["data_seed"])
    entries_b = sorted(entries_b, key=lambda e: e["data_seed"])
    assert len(entries_a) == len(entries_b)
    n_seeds = len(entries_a)

    acc_deg_a  = [np.array(e["per_item_acc_degraded"]) for e in entries_a]
    acc_null_a = [np.array(e["per_item_acc_null"])      for e in entries_a]
    jsd_a      = [np.array(e["per_item_jsd"])           for e in entries_a]
    acc_deg_b  = [np.array(e["per_item_acc_degraded"]) for e in entries_b]
    acc_null_b = [np.array(e["per_item_acc_null"])      for e in entries_b]
    jsd_b      = [np.array(e["per_item_jsd"])           for e in entries_b]

    n_items = acc_null_a[0].shape[0]
    rng_p = np.random.default_rng(seed)

    boot_diffs = []
    for _ in range(n_boot):
        # resample seeds with replacement, then items within each seed
        chosen = rng_p.integers(0, n_seeds, size=n_seeds)
        mi_a_list, mi_b_list = [], []
        for s_idx in chosen:
            idx = rng_p.integers(0, n_items, size=n_items)

            def compute_mi(acc_deg_s, acc_null_s, jsd_s, idx):
                U_b = utility(np.mean(acc_deg_s[:, idx], axis=1),
                              float(np.mean(acc_null_s[idx])))
                s_b = utility_crossing(U_b, S_GRID, window=3)
                if s_b is None:
                    return None
                R_b  = reliance(np.mean(jsd_s[:, idx], axis=1))
                rh_b = normalized_reliance(R_b)
                return misalignment_index(rh_b, S_GRID, s_b)

            ma = compute_mi(acc_deg_a[s_idx], acc_null_a[s_idx], jsd_a[s_idx], idx)
            mb = compute_mi(acc_deg_b[s_idx], acc_null_b[s_idx], jsd_b[s_idx], idx)
            if ma is not None and mb is not None:
                mi_a_list.append(ma)
                mi_b_list.append(mb)

        if mi_a_list:
            boot_diffs.append(float(np.mean(mi_a_list)) - float(np.mean(mi_b_list)))

    mean_diff = float(np.mean(boot_diffs)) if boot_diffs else float("nan")
    lo, hi = pct_ci(boot_diffs)
    return mean_diff, lo, hi, boot_diffs


# Per-seed MI values
fixed_mi_seeds    = {s: mi for s, mi in get_per_seed_mi("FixedModel")}
partial_mi_seeds  = {s: mi for s, mi in get_per_seed_mi("PartialGatedModel")}
gated_mi_seeds    = {s: mi for s, mi in get_per_seed_mi("GatedModel")}

seeds = sorted(fixed_mi_seeds.keys())
print("\nPer-seed MI (null='removed'):")
print(f"{'seed':>5} {'Fixed':>10} {'Partial':>10} {'Gated':>10} | {'Fixed>Partial':>13} {'Partial>Gated':>13}")
fp_count, pg_count = 0, 0
for s in seeds:
    f_mi = fixed_mi_seeds[s]
    p_mi = partial_mi_seeds[s]
    g_mi = gated_mi_seeds[s]
    fv = f"{f_mi:.4f}" if f_mi is not None else "None"
    pv = f"{p_mi:.4f}" if p_mi is not None else "None"
    gv = f"{g_mi:.4f}" if g_mi is not None else "None"
    fp_ok = (f_mi is not None and p_mi is not None and f_mi > p_mi)
    pg_ok = (p_mi is not None and g_mi is not None and p_mi > g_mi)
    if fp_ok: fp_count += 1
    if pg_ok: pg_count += 1
    print(f"{s:>5} {fv:>10} {pv:>10} {gv:>10} | {'YES':>13} {'YES' if pg_ok else 'NO':>13}")
print(f"\nFixed > Partial: {fp_count}/{len(seeds)} seeds")
print(f"Partial > Gated: {pg_count}/{len(seeds)} seeds")

# Paired bootstrap differences
fixed_entries   = records_by_group[("FixedModel",         "removed")]
partial_entries = records_by_group[("PartialGatedModel",  "removed")]
gated_entries   = records_by_group[("GatedModel",         "removed")]

print("\nRunning paired bootstrap MI(Fixed)-MI(Partial) …")
diff_fp_mean, diff_fp_lo, diff_fp_hi, _ = paired_boot_diff(fixed_entries, partial_entries)
print(f"  MI(Fixed)-MI(Partial): mean={diff_fp_mean:.4f}  95% CI={fmt_ci(diff_fp_lo, diff_fp_hi)}")

print("Running paired bootstrap MI(Partial)-MI(Gated) …")
diff_pg_mean, diff_pg_lo, diff_pg_hi, _ = paired_boot_diff(partial_entries, gated_entries)
print(f"  MI(Partial)-MI(Gated): mean={diff_pg_mean:.4f}  95% CI={fmt_ci(diff_pg_lo, diff_pg_hi)}")

# ──────────────────────────────────────────────────────────────────────────────
# Task 4 -- Evaluate success criteria (pre-declared, null='removed')
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("TASK 4 -- Success criterion verdicts (null='removed')")
print("=" * 100)

rem_df = df[df["null_kind"] == "removed"].set_index("model")

def get_mi(model): return rem_df.loc[model, "MI"]
def get_mi_hier_lo(model): return rem_df.loc[model, "MI_hier_lo"]
def get_mi_hier_hi(model): return rem_df.loc[model, "MI_hier_hi"]

mi_oracle  = get_mi("OracleGatedModel")
mi_fixed   = get_mi("FixedModel")
mi_partial = get_mi("PartialGatedModel")
mi_gated   = get_mi("GatedModel")

# Criterion (a): MI(oracle) < 0.05
crit_a = (not math.isnan(mi_oracle)) and mi_oracle < 0.05
print(f"\n(a) MI(oracle) < 0.05:   MI={mi_oracle:.4f}  ->  {'PASS' if crit_a else 'FAIL'}")

# Criterion (b): MI(fixed) > 0.7
crit_b = (not math.isnan(mi_fixed)) and mi_fixed > 0.7
print(f"(b) MI(fixed) > 0.7:     MI={mi_fixed:.4f}  ->  {'PASS' if crit_b else 'FAIL'}")

# Criterion (c) point-ordering: gated < partial < fixed
c_order = (not any(math.isnan(v) for v in [mi_gated, mi_partial, mi_fixed])
           and mi_gated < mi_partial < mi_fixed)
# Non-overlapping hierarchical CIs
def no_overlap(lo_a, hi_a, lo_b, hi_b):
    if any(v is None for v in [lo_a, hi_a, lo_b, hi_b]):
        return False
    return hi_a < lo_b or hi_b < lo_a

gated_lo,   gated_hi   = get_mi_hier_lo("GatedModel"),        get_mi_hier_hi("GatedModel")
partial_lo, partial_hi = get_mi_hier_lo("PartialGatedModel"),  get_mi_hier_hi("PartialGatedModel")
fixed_lo,   fixed_hi   = get_mi_hier_lo("FixedModel"),         get_mi_hier_hi("FixedModel")

gp_no_overlap = no_overlap(gated_lo, gated_hi, partial_lo, partial_hi)
pf_no_overlap = no_overlap(partial_lo, partial_hi, fixed_lo, fixed_hi)
crit_c_hier   = c_order and gp_no_overlap and pf_no_overlap

print(f"(c) Gated < Partial < Fixed with non-overlapping hier CIs:")
print(f"    Point ordering: {mi_gated:.4f} < {mi_partial:.4f} < {mi_fixed:.4f}  ->  {'OK' if c_order else 'FAIL'}")
print(f"    Gated hier CI:   {fmt_ci(gated_lo, gated_hi)}")
print(f"    Partial hier CI: {fmt_ci(partial_lo, partial_hi)}")
print(f"    Fixed hier CI:   {fmt_ci(fixed_lo, fixed_hi)}")
print(f"    No-overlap Gated vs Partial: {gp_no_overlap}")
print(f"    No-overlap Partial vs Fixed: {pf_no_overlap}")
print(f"    ->  {'PASS' if crit_c_hier else 'FAIL'}")

# Criterion (c) paired
fp_positive = diff_fp_lo is not None and diff_fp_lo > 0.0
pg_positive = diff_pg_lo is not None and diff_pg_lo > 0.0
crit_c_paired = fp_positive and pg_positive
print(f"(c) Paired: MI(Fixed)-MI(Partial) CI={fmt_ci(diff_fp_lo, diff_fp_hi)} -> lo>0: {fp_positive}")
print(f"            MI(Partial)-MI(Gated) CI={fmt_ci(diff_pg_lo, diff_pg_hi)} -> lo>0: {pg_positive}")
print(f"            ->  {'PASS' if crit_c_paired else 'FAIL'}")

# Criterion (d): MI non-increasing in gamma_scale
sharp_models = [f"SharpGatedModel_{g}" for g in ["0.5", "1.0", "2.0", "4.0", "8.0", "16.0"]]
sharp_mi = []
for m in sharp_models:
    if m in rem_df.index:
        v = rem_df.loc[m, "MI"]
        sharp_mi.append((m, v))
    else:
        sharp_mi.append((m, float("nan")))

print(f"\n(d) MI non-increasing in gamma_scale:")
prev_mi = None
crit_d = True
for mname, mval in sharp_mi:
    gamma = mname.split("_")[-1]
    flag = ""
    if prev_mi is not None and not math.isnan(mval) and mval > prev_mi + 1e-6:
        flag = "  ← VIOLATION"
        crit_d = False
    print(f"    γ×{gamma:>5}: MI={fmt_val(mval)}{flag}")
    if not math.isnan(mval):
        prev_mi = mval
print(f"    ->  {'PASS' if crit_d else 'FAIL'}")

print("\n" + "-" * 50)
overall = all([crit_a, crit_b, crit_c_hier, crit_c_paired, crit_d])
labels = [
    ("(a) oracle MI < 0.05", crit_a),
    ("(b) fixed MI > 0.7",   crit_b),
    ("(c) ordering+hier CI", crit_c_hier),
    ("(c) ordering+paired",  crit_c_paired),
    ("(d) gamma monotone",   crit_d),
]
for label, result in labels:
    print(f"  {label:30s}: {'PASS' if result else 'FAIL'}")
print(f"\n  Overall: {'ALL PASS' if overall else 'SOME FAIL'}")

# ──────────────────────────────────────────────────────────────────────────────
# Task 4 -- Figures
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("TASK 4 -- Generating figures …")

MODELS_MAIN = ["FixedModel", "GatedModel", "PartialGatedModel", "OracleGatedModel"]
COLORS = {
    "FixedModel":         "#d62728",
    "GatedModel":         "#2ca02c",
    "PartialGatedModel":  "#ff7f0e",
    "OracleGatedModel":   "#1f77b4",
}
LABELS = {
    "FixedModel":         "Fixed (quality-blind)",
    "GatedModel":         "Gated (quality-aware)",
    "PartialGatedModel":  "Partial-Gated",
    "OracleGatedModel":   "Oracle (control)",
}


def load_pooled_curves(model_name, null_kind="removed"):
    """Reconstruct U, R_hat from pooled data for a (model, null) pair."""
    entries = records_by_group.get((model_name, null_kind), [])
    if not entries:
        return None, None, None
    acc_deg_pool  = np.concatenate([np.array(e["per_item_acc_degraded"]) for e in entries], axis=1)
    acc_null_pool = np.concatenate([np.array(e["per_item_acc_null"]) for e in entries])
    jsd_pool      = np.concatenate([np.array(e["per_item_jsd"]) for e in entries], axis=1)
    U = utility(np.mean(acc_deg_pool, axis=1), float(np.mean(acc_null_pool)))
    R = reliance(np.mean(jsd_pool, axis=1))
    R_hat = normalized_reliance(R)
    return U, R_hat, smooth(U, window=3)


# ── Figure 1: Utility curves for main 4 models (null=removed)
fig1, ax1 = plt.subplots(figsize=(7, 4))
for m in MODELS_MAIN:
    U, R_hat, U_sm = load_pooled_curves(m, "removed")
    if U is None:
        continue
    ax1.plot(S_GRID, U_sm, color=COLORS[m], label=LABELS[m], lw=2)
ax1.axhline(0, color="k", lw=0.8, ls="--")
ax1.set_xlabel("Degradation severity S")
ax1.set_ylabel("Utility U(S)")
ax1.set_title("Fig 1: Utility curves (null = removed, pooled)")
ax1.legend(fontsize=9)
ax1.set_xlim(0, 1)
fig1.tight_layout()
fig1.savefig(FIG_DIR / "fig1_utility_curves_v04.pdf")
plt.close(fig1)

# ── Figure 2: Reliance (R_hat) curves for main 4 models (null=removed)
fig2, ax2 = plt.subplots(figsize=(7, 4))
for m in MODELS_MAIN:
    U, R_hat, U_sm = load_pooled_curves(m, "removed")
    if R_hat is None:
        continue
    ax2.plot(S_GRID, R_hat, color=COLORS[m], label=LABELS[m], lw=2)
ax2.axhline(1.0, color="k", lw=0.8, ls="--", label="R_hat = 1")
ax2.set_xlabel("Degradation severity S")
ax2.set_ylabel(r"Normalised reliance $\hat{R}(S)$")
ax2.set_title("Fig 2: Normalised reliance curves (null = removed, pooled)")
ax2.legend(fontsize=9)
ax2.set_xlim(0, 1)
fig2.tight_layout()
fig2.savefig(FIG_DIR / "fig2_reliance_curves_v04.pdf")
plt.close(fig2)

# ── Figure 3: MI bar chart with hier CI error bars (null=removed)
fig3, ax3 = plt.subplots(figsize=(8, 4))
bar_models = MODELS_MAIN + [f"SharpGatedModel_{g}" for g in ["0.5","1.0","2.0","4.0","8.0","16.0"]]
x = np.arange(len(bar_models))
mis, errs_lo, errs_hi = [], [], []
for m in bar_models:
    if m not in rem_df.index:
        mis.append(float("nan")); errs_lo.append(0); errs_hi.append(0)
        continue
    row = rem_df.loc[m]
    mi_v = row["MI"]
    lo = row["MI_hier_lo"] if row["MI_hier_lo"] is not None else mi_v
    hi = row["MI_hier_hi"] if row["MI_hier_hi"] is not None else mi_v
    mis.append(mi_v if not math.isnan(mi_v) else 0)
    errs_lo.append(max(0, mi_v - lo) if not math.isnan(mi_v) else 0)
    errs_hi.append(max(0, hi - mi_v) if not math.isnan(mi_v) else 0)

bar_colors = [COLORS.get(m, "#9467bd") for m in bar_models]
ax3.bar(x, mis, color=bar_colors, alpha=0.75)
ax3.errorbar(x, mis, yerr=[errs_lo, errs_hi], fmt="none", color="black", capsize=4, lw=1.5)
ax3.axhline(1.0, color="k", lw=0.8, ls="--", label="MI = 1")
ax3.set_xticks(x)
short_labels = [
    m.replace("SharpGatedModel_", "Sharp\n×").replace("GatedModel", "Gated")
     .replace("PartialGatedModel", "Partial").replace("FixedModel", "Fixed")
     .replace("OracleGatedModel", "Oracle")
    for m in bar_models
]
ax3.set_xticklabels(short_labels, fontsize=8)
ax3.set_ylabel("Misalignment Index (unclipped)")
ax3.set_title("Fig 3: MI with hierarchical CI (null = removed)")
ax3.legend(fontsize=9)
fig3.tight_layout()
fig3.savefig(FIG_DIR / "fig3_mi_bar_v04.pdf")
plt.close(fig3)

# ── Figure 4: MI vs gamma_scale (SharpGatedModel sweep) with hier CI
gamma_vals  = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
gamma_names = [f"SharpGatedModel_{g}" for g in gamma_vals]
g_mis, g_lo, g_hi = [], [], []
for m in gamma_names:
    if m not in rem_df.index:
        g_mis.append(float("nan")); g_lo.append(float("nan")); g_hi.append(float("nan"))
        continue
    row = rem_df.loc[m]
    g_mis.append(row["MI"])
    g_lo.append(row["MI_hier_lo"] if row["MI_hier_lo"] is not None else float("nan"))
    g_hi.append(row["MI_hier_hi"] if row["MI_hier_hi"] is not None else float("nan"))

g_mis = np.array(g_mis, dtype=float)
g_lo  = np.array(g_lo,  dtype=float)
g_hi  = np.array(g_hi,  dtype=float)

fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.plot(gamma_vals, g_mis, "o-", color="#9467bd", lw=2, label="MI (point)")
ax4.fill_between(gamma_vals, g_lo, g_hi, alpha=0.25, color="#9467bd", label="95% hier CI")
# Also show GatedModel (γ×1 baseline) as horizontal reference
if not math.isnan(mi_gated):
    ax4.axhline(mi_gated, color=COLORS["GatedModel"], ls="--", lw=1.2,
                label=f"GatedModel MI={mi_gated:.3f}")
ax4.set_xscale("log")
ax4.set_xticks(gamma_vals)
ax4.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax4.set_xlabel("gamma_scale (log scale)")
ax4.set_ylabel("Misalignment Index (unclipped)")
ax4.set_title("Fig 4: MI vs. gamma_scale -- SharpGatedModel sweep (null = removed)")
ax4.legend(fontsize=9)
fig4.tight_layout()
fig4.savefig(FIG_DIR / "fig4_gamma_sweep_v04.pdf")
plt.close(fig4)

print(f"Figures saved to {FIG_DIR}/fig{{1-4}}_*_v04.pdf")
print("\nDone.")
