"""Small command-line interface for the study's independent workflows."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def execute(command: list[str]) -> int:
    """Preserve child-process exit codes and show the exact command executed."""
    print("Running:", subprocess.list2cmdline(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.call(command, cwd=ROOT, env=environment)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def integrity() -> int:
    manifest = ROOT / "reproducibility" / "SHA256SUMS.json"
    if not manifest.is_file():
        print("The repository integrity manifest is missing.", file=sys.stderr)
        return 1
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    problems = []
    for name, expected in entries.items():
        path = ROOT / name
        if not path.is_file():
            problems.append({"file": name, "problem": "missing"})
        elif sha256(path) != expected:
            problems.append({"file": name, "problem": "checksum differs"})
    print(json.dumps({"status": "PASS" if not problems else "FAIL", "files_checked": len(entries), "problems": problems}, indent=2))
    return int(bool(problems))


def doctor(training: bool = False) -> int:
    report = {"python": sys.version.split()[0], "python_supported": sys.version_info >= (3, 10),
              "reference_predictions_present": (ROOT / "data/reference/main_region_predictions.csv.gz").is_file(),
              "Rscript_available": shutil.which("Rscript") is not None}
    missing = []
    if training:
        packages = ["torch", "ultralytics", "monai", "nnunetv2", "dynamic-network-architectures", "numpy", "pandas", "Pillow", "PyYAML", "scipy", "scikit-learn"]
        versions = {}
        for name in packages:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
                missing.append(name)
        report["training_package_versions"] = versions
        report["missing_training_packages"] = missing
        if "torch" not in missing:
            import torch
            report["cuda_available"] = bool(torch.cuda.is_available())
            report["cuda_runtime"] = torch.version.cuda
    failed = not report["python_supported"] or not report["reference_predictions_present"] or bool(missing)
    if training and report.get("cuda_available") is False:
        failed = True
    report["status"] = "FAIL" if failed else "PASS"
    print(json.dumps(report, indent=2))
    return int(failed)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Let each model workflow own its own arguments and help text.
    if argv and argv[0] in {"main", "benchmark", "additional"}:
        script = ROOT / "pipelines" / argv[0] / "reproduce.py"
        if not script.is_file():
            print(f"Missing workflow entry point: {script.relative_to(ROOT)}", file=sys.stderr)
            return 1
        return execute([sys.executable, str(script), *argv[1:]])
    parser = argparse.ArgumentParser(description="Reproduce proximal caries training, assessment, performance metrics and statistics.")
    sub = parser.add_subparsers(dest="command", required=True)
    reproduce = sub.add_parser("reproduce", help="Recompute the main clinical tables from included regional predictions; no extra packages or GPU")
    reproduce.add_argument("--input", type=Path, default=ROOT / "data/reference/main_region_predictions.csv.gz")
    reproduce.add_argument("--out", type=Path, default=ROOT / "results/descriptive")
    reproduce.add_argument("--no-reference-check", action="store_true", help="Do not compare against the supplied published summaries")
    statistics = sub.add_parser("statistics", help="Run the six paired detection-accuracy comparisons in R")
    statistics.add_argument("--input", type=Path, default=ROOT / "data/reference/main_region_predictions.csv.gz")
    statistics.add_argument("--out", type=Path, default=ROOT / "results/statistics")
    statistics.add_argument("--rscript", default="Rscript", help="Rscript executable name or path")
    statistics.add_argument("--no-reference-check", action="store_true", help="Analyse new model predictions without comparing their effects with the original published results")
    status = sub.add_parser("doctor", help="Check Python, reference files and optional training dependencies")
    status.add_argument("--training", action="store_true")
    sub.add_parser("verify", help="Check every released source, configuration and reference-data checksum")
    datasets = sub.add_parser("datasets", help="Inspect or assemble downloaded datasets without changing image pixels")
    data_sub = datasets.add_subparsers(dest="data_command", required=True)
    inspect = data_sub.add_parser("inspect", help="Inspect dataset folder names and required files")
    inspect.add_argument("archive", type=Path)
    assemble = data_sub.add_parser("assemble", help="Match downloaded files to the frozen study inventory by checksum")
    assemble.add_argument("--inputs", type=Path, nargs="+", help="Downloaded ZIPs or directories; defaults to data/downloads plus the included study annotations")
    assemble.add_argument("--out", type=Path, default=ROOT / "data/Datasets.zip")
    assemble.add_argument("--annotation-version", choices=["recorded", "updated"], default="updated")
    sub.add_parser("main", help="Prepare, train and assess the eight main strategies; see main --help")
    sub.add_parser("benchmark", help="Prepare, train and assess the architecture comparison; see benchmark --help")
    sub.add_parser("additional", help="Run the additional experiments described in Appendix B; see additional --help")
    args = parser.parse_args(argv)
    if args.command == "reproduce":
        command = [sys.executable, str(ROOT / "analysis/reproduce_descriptive.py"), "--input", str(args.input.resolve()), "--out", str(args.out.resolve())]
        if not args.no_reference_check:
            command.append("--verify-reference")
        return execute(command)
    if args.command == "statistics":
        executable = shutil.which(args.rscript)
        if executable is None:
            print("Rscript was not found. Install R and run Rscript analysis/install_packages.R, or supply --rscript with its executable path.", file=sys.stderr)
            return 2
        command = [executable, str(ROOT / "analysis/detection_accuracy.R"), str(args.input.resolve()), str(args.out.resolve())]
        if args.no_reference_check:
            command.append("--no-reference-check")
        return execute(command)
    if args.command == "doctor":
        return doctor(args.training)
    if args.command == "verify":
        return integrity()
    if args.command == "datasets":
        from .datasets import assemble_archives, inspect_archive
        if args.data_command == "inspect":
            print(json.dumps(inspect_archive(args.archive, include_members=False), indent=2))
            return 0
        inputs = list(args.inputs or [ROOT / "data/downloads"])
        supplement = ROOT / "data/study_annotations.zip"
        if args.annotation_version == "updated" and supplement.is_file() and supplement.resolve() not in {p.resolve() for p in inputs}:
            inputs.append(supplement)
        name = "data_fingerprints_updated.json" if args.annotation_version == "updated" else "data_fingerprints.json"
        try:
            report = assemble_archives(inputs, args.out, ROOT / "configs/main" / name)
        except (ValueError, FileNotFoundError) as error:
            print(str(error), file=sys.stderr)
            if hasattr(error, "missing_files"):
                print(json.dumps({"missing_files": error.missing_files}, indent=2), file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2))
        return 0
    return 1
