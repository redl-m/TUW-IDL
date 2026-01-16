#!/usr/bin/env python3

import glob
from pathlib import Path
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

from utils.config import EMOTION_MAP, LABEL2ID, TRAIN_ACTORS, VAL_ACTORS, TEST_ACTORS
from utils.paths import DATA_PROCESSED_DIR


class BaseRAVDESSDataset(Dataset, ABC):
    def __init__(self, split: str = "train", root_dir: Path = DATA_PROCESSED_DIR, subset_dir: str ="", ext: str = ""):
        self.split = split
        self.ext = ext
        self.root_dir = Path(root_dir) / subset_dir
        self.files = self._load_metadata()

    def _load_metadata(self):
        # Determine file extension based on subset
        search_pattern = str(self.root_dir / "**" / f"*{self.ext}")
        files = glob.glob(search_pattern, recursive=True)

        valid_files = []
        for f in files:
            path = Path(f)
            parts = path.name.split('-')

            # Extract identifiers
            actor_id = int(parts[6].split('.')[0])
            emotion_code = parts[2]

            # Filter by Split
            if self.split == "train" and actor_id not in TRAIN_ACTORS: continue
            if self.split == "val" and actor_id not in VAL_ACTORS: continue
            if self.split == "test" and actor_id not in TEST_ACTORS: continue

            if emotion_code in EMOTION_MAP:
                valid_files.append(path)

        return valid_files

    def _get_label(self, file_path):
        parts = file_path.name.split('-')
        emotion_code = parts[2]
        return LABEL2ID[EMOTION_MAP[emotion_code]]

    def __len__(self):
        return len(self.files)

    @abstractmethod
    def __getitem__(self, idx):
        pass