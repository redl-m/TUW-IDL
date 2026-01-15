#!/usr/bin/env python3

import kagglehub
import shutil
import os

# Download to kagglehub cache
download_path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

# Define target directory
target_dir = "data/raw"
os.makedirs(target_dir, exist_ok=True)

# Move files from cache to local folder
for item in os.listdir(download_path):
    s = os.path.join(download_path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print(f"Dataset successfully moved to: {os.path.abspath(target_dir)}")