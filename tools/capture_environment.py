#!/usr/bin/env python3
"""Save package and hardware versions alongside a reproduction run."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("results/environment.json"))
args = parser.parse_args()
packages = ["torch", "ultralytics", "monai", "nnunetv2", "dynamic-network-architectures", "numpy", "pandas", "Pillow", "PyYAML", "scipy", "scikit-learn", "scikit-image", "matplotlib", "seaborn", "batchgenerators", "batchgeneratorsv2", "blosc2"]
versions = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
report = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
          "system": platform.system(), "machine": platform.machine(), "packages": versions}
if versions["torch"] is not None:
    import torch
    report["cuda_runtime"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(args.out)
