#!/usr/bin/env python3

import os

from src.utils.paths import DATA_RAW_DIR, ensure_paths

import kagglehub
import shutil

ensure_paths([DATA_RAW_DIR])

# Download to kagglehub cache
download_path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
download_path = os.path.join(download_path, "audio_speech_actors_01-24")

# Move files from cache to local folder
for item in os.listdir(download_path):
    s = os.path.join(download_path, item)
    d = os.path.join(DATA_RAW_DIR, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print(f"Dataset successfully moved to: {os.path.abspath(DATA_RAW_DIR)}")