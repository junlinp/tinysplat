#!/usr/bin/env python3
"""Checkout entry point for tinysplat.pinhole_training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))

from tinysplat import pinhole_training as _training

# Keep imports from the original script working for downstream tools.
__all__ = [name for name in vars(_training) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_training, name)


if __name__ == "__main__":
    _training.main()
