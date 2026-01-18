#!/usr/bin/env python3

from loaders.base import BaseRAVDESSDataset

import torch
import numpy as np

from typing import Any, Dict


class FeatureDataset(BaseRAVDESSDataset):
    """
    Dataset class for loading pre-extracted audio features.

    This class loads .npy files from DATA_PROCESSED_DIR/AudioFeatureExtractor containing
    features vector of shape (3, n_features), storing a clean, noisy and augmented version
    of the features. During training one of the three is randomly sampled, while during
    testing the clean one is always used.
    """
    def __init__(self, split: str = "train"):
        """
        Initializes the FeatureDataset class.
        """
        super().__init__(split=split, subset_dir="AudioFeatureExtractor", ext=".npy")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.files[idx]

        # Shape: (3, N_Features) -> [Clean, Noise, Aug]
        features = np.load(file_path)

        if self.split == "train":
            row_idx = np.random.randint(0, 3)
            selected_features = features[row_idx]
        else:
            selected_features = features[0]

        return {
            "features": torch.tensor(selected_features, dtype=torch.float32),
            "labels": torch.tensor(self._get_label(file_path), dtype=torch.long)
        }