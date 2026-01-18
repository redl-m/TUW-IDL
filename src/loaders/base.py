#!/usr/bin/env python3

from utils.config import EMOTION_MAP, LABEL2ID, TRAIN_ACTORS, VAL_ACTORS, TEST_ACTORS, DATA_PROCESSED_DIR

import glob
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List

from torch.utils.data import Dataset


class BaseRAVDESSDataset(Dataset, ABC):
    """
    Abstract Base Class for RAVDESS datasets.

    This class handles the logic for parsing the specific RAVDESS filename
    structure, where each number represents certain information:

    03-01-06-01-02-01-12.wav
    Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor

    Also, this class can be used to split up the dataset into train, val and test,
    as configured in config.
    """
    def __init__(self, split: str = "train", root_dir: Path = DATA_PROCESSED_DIR, subset_dir: str = "", ext: str = "") -> None:
        """
        Initializes the BaseRAVDESSDataset class.
        """
        self.split = split
        self.ext = ext
        self.root_dir = Path(root_dir) / subset_dir
        self.files = self._load_metadata()

    def _load_metadata(self) -> List[Path]:
        """
        Scans subdirectory for paths with a certain extensions, specified by classes
        that inherit from this class, and returns a list of valid file paths.
        """
        search_pattern = str(self.root_dir / "**" / f"*{self.ext}")
        files = glob.glob(search_pattern, recursive=True)

        # RAVDESS file name convention, see above
        valid_files = []
        for f in files:
            path = Path(f)
            parts = path.name.split('-')

            if len(parts) < 7:
                continue # Skip files not matching this convention

            # Extract identifiers
            actor_id = int(parts[6].split('.')[0])
            emotion_code = parts[2]

            # Filter by Split
            if self.split == "train" and actor_id not in TRAIN_ACTORS: continue
            if self.split == "val" and actor_id not in VAL_ACTORS: continue
            if self.split == "test" and actor_id not in TEST_ACTORS: continue

            # Ensure emotion code exists
            if emotion_code in EMOTION_MAP:
                valid_files.append(path)

        return valid_files

    def _get_label(self, file_path: Path) -> int:
        """
        Extracts the label ID from the file path.
        """
        parts = file_path.name.split('-')
        emotion_code = parts[2]
        return LABEL2ID[EMOTION_MAP[emotion_code]]

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.files)

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        Retrieve and process a single audio sample.
        """
        pass