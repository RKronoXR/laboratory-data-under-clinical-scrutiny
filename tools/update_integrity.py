#!/usr/bin/env python3
"""Rebuild the source-release checksum inventory after an intentional edit."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = root / "reproducibility/SHA256SUMS.json"
excluded = {".git", ".venv", "venv", "runs", "results", "build", "dist", "__pycache__", ".pytest_cache"}
manifest = {}
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root)
    if not path.is_file() or path == destination or any(p in excluded or p.endswith(".egg-info") for p in relative.parts):
        continue
    if path.is_symlink() or path.suffix in {".pyc", ".pyo"}:
        continue
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    manifest[relative.as_posix()] = h.hexdigest()
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Recorded {len(manifest)} file checksums.")
