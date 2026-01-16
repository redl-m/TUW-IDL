#!/usr/bin/env python3

from loaders.base import BaseRAVDESSDataset

import torch
import numpy as np


class FeatureDataset(BaseRAVDESSDataset):
    def __init__(self, split="train"):
        super().__init__(split=split, subset_dir="AudioFeatureExtractor", ext=".npy")

    def __getitem__(self, idx):
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