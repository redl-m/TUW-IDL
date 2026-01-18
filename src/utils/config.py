#!/usr/bin/env python3

"""
Configuration file for constants, paths and more.
"""

import os
from pathlib import Path


EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

LABEL2ID = {v: i for i, v in enumerate(EMOTION_MAP.values())}
ID2LABEL = {i: v for v, i in LABEL2ID.items()}

# Data split
TRAIN_ACTORS = set(range(1, 21))
VAL_ACTORS = set(range(21, 23))
TEST_ACTORS = set(range(23, 25))

# Model configuration
BASE_MODEL = "facebook/wav2vec2-base"
NUM_LABELS = 8

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
CACHE_DIR = ROOT_DIR / "cache"

def ensure_paths(paths: list[Path]) -> None:
    """
    Ensures that all paths in the provided list exist.
    """
    for d in paths:
        os.makedirs(d, exist_ok=True)