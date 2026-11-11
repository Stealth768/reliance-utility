#!/usr/bin/env python3
"""Package extended paper files and figures into a self-contained zip bundle for Overleaf."""

import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def build_bundle():
    zip_dest = REPO_ROOT / "comsia2026_extended_overleaf.zip"
    downloads_dest = Path.home() / "Downloads" / "comsia2026_extended_overleaf.zip"
    desktop_dest = Path.home() / "Desktop" / "comsia2026_extended_overleaf.zip"

    src_dir = REPO_ROOT / "paper_extended"

    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. Essential root files for 1-click Overleaf compile
        for fname in ["main.tex", "refs.bib", "svmult.cls", "spmpsci.bst", "spbasic.bst", "spphys.bst", "svind.ist"]:
            p = src_dir / fname
            if p.exists():
                z.write(p, arcname=fname)

        # 2. Add figures/ folder
        figs_dir = src_dir / "figures"
        if figs_dir.exists():
            for p in figs_dir.rglob("*"):
                if p.is_file() and not p.name.startswith("."):
                    arc = p.relative_to(src_dir).as_posix()
                    z.write(p, arcname=arc)

    size = zip_dest.stat().st_size
    print(f"Created bundle at: {zip_dest} ({size} bytes)")

    try:
        shutil.copy2(zip_dest, downloads_dest)
        print(f"Copied bundle to: {downloads_dest} ({downloads_dest.stat().st_size} bytes)")
    except Exception as e:
        print(f"Failed to copy to Downloads: {e}")

    try:
        shutil.copy2(zip_dest, desktop_dest)
        print(f"Copied bundle to: {desktop_dest} ({desktop_dest.stat().st_size} bytes)")
    except Exception as e:
        print(f"Failed to copy to Desktop: {e}")

if __name__ == "__main__":
    build_bundle()
