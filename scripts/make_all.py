"""Regenerate all derived tables and figures from raw results.

Execution order:
1. scripts/analyze_vlm_all.py (VLM main benchmarks, tables, curves)
2. scripts/make_synthetic_figs.py (Synthetic figures 1-4)
3. scripts/hash_results.py (SHA256 manifest of all raw results)

Exits nonzero if any step fails.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_step(step_name: str, cmd: list):
    print(f"\n==================================================")
    print(f"RUNNING STEP: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"==================================================")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print(f"[ERROR] Step '{step_name}' failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[OK] Step '{step_name}' completed successfully.")


def main():
    py = sys.executable

    # 1. Analyze VLM benchmark
    run_step(
        "Analyze VLM Benchmark",
        [py, "scripts/analyze_vlm_all.py"],
    )

    # 2. Make synthetic figures
    run_step(
        "Make Synthetic Figures",
        [py, "scripts/make_synthetic_figs.py"],
    )

    # 3. Hash raw results
    run_step(
        "Hash Raw Results",
        [py, "scripts/hash_results.py"],
    )

    print("\n[SUCCESS] make_all completed all steps successfully.")


if __name__ == "__main__":
    main()
