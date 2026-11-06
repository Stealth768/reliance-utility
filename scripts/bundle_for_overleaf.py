"""Package paper files, tables, and figures into a zip bundle for Overleaf."""
import zipfile
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def build_bundle():
    zip_dest = REPO_ROOT / "overleaf_springer.zip"
    downloads_dest = Path(r"C:\Users\steal\Downloads\overleaf_springer.zip")

    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. Root files for easy compilation
        z.write(REPO_ROOT / "paper" / "main.tex", arcname="main.tex")
        z.write(REPO_ROOT / "paper" / "svmult.cls", arcname="svmult.cls")
        z.write(REPO_ROOT / "paper" / "spmpsci.bst", arcname="spmpsci.bst")

        # 2. Add paper/ directory
        for p in (REPO_ROOT / "paper").rglob("*"):
            if p.is_file() and not p.name.endswith(".pyc") and not p.name == "appendix.tex":
                z.write(p, arcname=p.relative_to(REPO_ROOT).as_posix())

        # 3. Add tables/ directory
        for p in (REPO_ROOT / "tables").rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(REPO_ROOT).as_posix())

        # 4. Add figures/ directory
        for p in (REPO_ROOT / "figures").rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(REPO_ROOT).as_posix())

    print(f"Created bundle at: {zip_dest} ({zip_dest.stat().st_size} bytes)")
    shutil.copy2(zip_dest, downloads_dest)
    print(f"Copied bundle to: {downloads_dest} ({downloads_dest.stat().st_size} bytes)")

if __name__ == "__main__":
    build_bundle()
