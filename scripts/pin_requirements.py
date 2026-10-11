"""Pin environment dependencies to requirements.lock.

Run this script inside the verified Kaggle GPU environment to capture exact
versions of all installed packages:

    python scripts/pin_requirements.py

This generates `requirements.lock` at the root of the repository.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def pin_environment(output_path: Path):
    print("Collecting current pip environment freeze...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        lock_content = (
            "# Auto-generated lockfile by scripts/pin_requirements.py\n"
            f"# Python: {sys.version.split()[0]} ({sys.platform})\n"
            "# Run inside Kaggle GPU environment for reproducible replication.\n\n"
            + result.stdout
        )
        output_path.write_text(lock_content, encoding="utf-8")
        print(f"Successfully wrote {len(result.stdout.splitlines())} pinned packages to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to run pip freeze: {e}")
        sys.exit(1)


def main():
    out_file = REPO_ROOT / "requirements.lock"
    pin_environment(out_file)


if __name__ == "__main__":
    main()
