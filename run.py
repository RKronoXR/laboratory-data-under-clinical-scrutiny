#!/usr/bin/env python3
"""Run the study workflows directly from a downloaded repository."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from carieslab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
