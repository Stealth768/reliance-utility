"""Verify numbers embedded in LaTeX documents against generated CSV tables.

Parses custom LaTeX macros of the form:
    \\checknum{value}{model:corruption:column}
or
    \\checknum{value}{label}

Verifies that the hard-coded number in paper/*.tex matches the generated
value in results/derived/vlm_main_summary.csv within numerical tolerance (1e-3).

Exits 0 if all numbers match or if paper/ does not contain checknum annotations.
Exits 1 if any discrepancy is found.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_checknums_from_tex(tex_file: Path) -> List[Tuple[str, str, int, Path]]:
    """Parse all \\checknum{value}{label} occurrences from a TeX file."""
    pattern = re.compile(r"\\checknum\{([^}]+)\}\{([^}]+)\}")
    matches = []
    try:
        content = tex_file.read_text(encoding="utf-8")
        for line_num, line in enumerate(content.splitlines(), start=1):
            for m in pattern.finditer(line):
                val_str = m.group(1).strip()
                label = m.group(2).strip()
                matches.append((val_str, label, line_num, tex_file))
    except Exception as e:
        print(f"[WARN] Failed to read {tex_file}: {e}")
    return matches


def check_numbers(
    paper_dir: Path,
    summary_csv: Path,
    tol: float = 1e-3,
) -> bool:
    """Compare checknum values with summary CSV table."""
    if not paper_dir.exists():
        print(f"[INFO] Paper directory '{paper_dir}' does not exist. Skipping check.")
        return True

    tex_files = list(paper_dir.glob("**/*.tex"))
    if not tex_files:
        print(f"[INFO] No .tex files found in '{paper_dir}'. Skipping check.")
        return True

    all_checks = []
    for tf in tex_files:
        all_checks.extend(parse_checknums_from_tex(tf))

    if not all_checks:
        print(f"[INFO] No \\checknum macros found in {len(tex_files)} .tex file(s). All good.")
        return True

    if not summary_csv.exists():
        print(f"[ERROR] Summary CSV '{summary_csv}' does not exist, but \\checknum macros were found.")
        return False

    df = pd.read_csv(summary_csv)
    # Build lookup index by (model_name, corruption)
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, row in df.iterrows():
        m_name = str(row.get("model_name", "")).strip()
        corr = str(row.get("corruption", "")).strip()
        lookup[(m_name, corr)] = dict(row)

    mismatches = 0
    total_checks = len(all_checks)

    print(f"Verifying {total_checks} \\checknum annotations against {summary_csv.name}...")

    for val_str, label, line_num, fpath in all_checks:
        parts = label.split(":")
        if len(parts) == 3:
            model_key, corr_key, col_key = parts[0].strip(), parts[1].strip(), parts[2].strip()
            # Match model key by exact or basename
            matched_row = None
            for (m, c), r in lookup.items():
                if (m == model_key or m.endswith(model_key) or model_key in m) and c == corr_key:
                    matched_row = r
                    break

            if matched_row is None:
                print(
                    f"  [FAIL] {fpath.name}:{line_num} - Condition '{model_key}:{corr_key}' not found in summary table."
                )
                mismatches += 1
                continue

            if col_key not in matched_row:
                print(
                    f"  [FAIL] {fpath.name}:{line_num} - Column '{col_key}' not found in summary table."
                )
                mismatches += 1
                continue

            actual_val = matched_row[col_key]
            try:
                exp_f = float(val_str)
                act_f = float(actual_val)
                diff = abs(exp_f - act_f)
                if diff > tol:
                    print(
                        f"  [FAIL] {fpath.name}:{line_num} - {label}: TeX has {val_str}, table has {actual_val} (diff={diff:.5f} > tol={tol})"
                    )
                    mismatches += 1
                else:
                    print(f"  [PASS] {fpath.name}:{line_num} - {label} = {val_str}")
            except ValueError:
                # String comparison
                if str(actual_val).strip() != val_str:
                    print(
                        f"  [FAIL] {fpath.name}:{line_num} - {label}: TeX has '{val_str}', table has '{actual_val}'"
                    )
                    mismatches += 1
                else:
                    print(f"  [PASS] {fpath.name}:{line_num} - {label} = {val_str}")
        else:
            print(f"  [WARN] {fpath.name}:{line_num} - Label '{label}' does not follow 'model:corr:column' format.")

    if mismatches > 0:
        print(f"\n[ERROR] Found {mismatches}/{total_checks} number mismatches between paper and table.")
        return False
    else:
        print(f"\n[OK] All {total_checks} numbers verified accurately against summary table.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Check numbers in LaTeX documents against CSV tables")
    parser.add_argument(
        "--paper_dir",
        type=str,
        default="paper",
        help="Directory containing LaTeX source files",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="results/derived/vlm_main_summary.csv",
        help="Path to generated summary CSV",
    )
    args = parser.parse_args()

    ok = check_numbers(Path(args.paper_dir), Path(args.summary_csv))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
