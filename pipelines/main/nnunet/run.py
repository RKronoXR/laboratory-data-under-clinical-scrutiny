#!/usr/bin/env python3
"""Private worker dispatcher for the public reproduction entry point."""
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "_worker":
        raise SystemExit("Use the public reproduce.py entry point. This file only executes a signed worker task.")
    from reversekd.worker import execute
    raise SystemExit(execute(sys.argv[2]))
