"""BEV (Bird's Eye View) Driving Study Feasibility Probe.

Audits whether the environment has prerequisites for running BEVFusion on nuScenes-mini:
- Python version, PyTorch version, CUDA availability
- MMDetection3D / MMCV / MMEngine installation
- Disk space availability (requires >= 25 GB free)
- Presence of nuScenes-mini dataset directory

DOES NOT download datasets or compile any packages.
Writes results/derived/bev_probe.json and prints GO / NO-GO verdict.
"""

import json
import os
import platform
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_pkg_import(name: str):
    """Check if package can be imported and return version or None."""
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "installed")
    except Exception:
        return None


def run_probe():
    reasons = []
    go = True

    # 1. Python info
    py_ver = sys.version.split()[0]
    print(f"Python Version : {py_ver} ({platform.system()} {platform.machine()})")

    # 2. PyTorch & CUDA
    torch_ver = check_pkg_import("torch")
    cuda_avail = False
    cuda_ver = "None"
    gpu_name = "None"

    if torch_ver is not None:
        import torch
        cuda_avail = torch.cuda.is_available()
        cuda_ver = getattr(torch.version, "cuda", "unknown") if cuda_avail else "n/a"
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "n/a"
        print(f"PyTorch Version: {torch_ver} (CUDA available: {cuda_avail}, CUDA: {cuda_ver}, Device: {gpu_name})")
    else:
        print("PyTorch Version: NOT INSTALLED")
        reasons.append("PyTorch is not installed in the current environment.")
        go = False

    if not cuda_avail:
        reasons.append("CUDA GPU accelerator is not available.")
        go = False

    # 3. MMDetection3D ecosystem
    mmcv_ver = check_pkg_import("mmcv")
    mmdet3d_ver = check_pkg_import("mmdet3d")
    mmengine_ver = check_pkg_import("mmengine")

    print(f"MMCV Version   : {mmcv_ver if mmcv_ver else 'NOT INSTALLED'}")
    print(f"MMDet3D Version: {mmdet3d_ver if mmdet3d_ver else 'NOT INSTALLED'}")
    print(f"MMEngine       : {mmengine_ver if mmengine_ver else 'NOT INSTALLED'}")

    if not mmcv_ver:
        reasons.append("mmcv is not installed (required for BEVFusion).")
        go = False
    if not mmdet3d_ver:
        reasons.append("mmdet3d is not installed (required for BEVFusion).")
        go = False

    # 4. Disk space check
    check_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    total, used, free = shutil.disk_usage(check_dir)
    free_gb = free / (1024 ** 3)
    print(f"Disk Free Space: {free_gb:.1f} GB at {check_dir}")

    if free_gb < 20.0:
        reasons.append(f"Insufficient free disk space ({free_gb:.1f} GB < 20.0 GB threshold for nuScenes-mini).")
        go = False

    # 5. nuScenes-mini existence check
    nuscenes_candidates = [
        Path("/kaggle/input/nuscenes-mini"),
        Path("/kaggle/working/nuscenes-mini"),
        Path("./data/nuscenes-mini"),
        Path("./nuscenes-mini"),
        REPO_ROOT / "data" / "nuscenes-mini",
    ]
    nuscenes_found = any(p.exists() for p in nuscenes_candidates)
    print(f"nuScenes-mini  : {'FOUND' if nuscenes_found else 'NOT FOUND'}")
    if not nuscenes_found:
        reasons.append("nuScenes-mini dataset directory not found locally or in /kaggle/input.")
        go = False

    verdict = "GO" if go else "NO-GO"

    print("\n" + "=" * 50)
    print(f"BEV FEASIBILITY VERDICT: {verdict}")
    print("=" * 50)
    if reasons:
        print("Reasons:")
        for r in reasons:
            print(f"  - {r}")

    probe_result = {
        "verdict": verdict,
        "python_version": py_ver,
        "platform": platform.platform(),
        "torch_version": torch_ver,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "gpu_name": gpu_name,
        "mmcv_version": mmcv_ver,
        "mmdet3d_version": mmdet3d_ver,
        "mmengine_version": mmengine_ver,
        "disk_free_gb": round(free_gb, 2),
        "nuscenes_found": nuscenes_found,
        "reasons": reasons,
    }

    out_file = REPO_ROOT / "results" / "derived" / "bev_probe.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(probe_result, f, indent=2)
    print(f"\nSaved probe summary to: {out_file}")

    return verdict


if __name__ == "__main__":
    run_probe()
