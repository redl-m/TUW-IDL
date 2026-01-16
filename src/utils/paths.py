#!/usr/bin/env python3

import os
from pathlib import Path

# define global paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
CACHE_DIR = ROOT_DIR / "cache"

def ensure_paths(paths: list[Path]) -> None:
    """
    Makes sure all paths exist.
    """
    for d in paths:
        os.makedirs(d, exist_ok=True)