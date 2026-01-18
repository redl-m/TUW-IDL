#!/usr/bin/env python3

from loaders.base import BaseRAVDESSDataset
from utils.config import BASE_MODEL, CACHE_DIR

import torch
from typing import Any, Dict
from transformers import Wav2Vec2FeatureExtractor
from audiomentations import Compose, AddGaussianNoise, PitchShift, Gain


class WaveformDataset(BaseRAVDESSDataset):
    """
    Dataset for loading raw waveforms.

    This class loads the pre-processed PyTorch tensor files (.pt) located in
    DATA_PROCESSED_DIR/AudioWaveformPreprocessor. On-the-fly augmentations
    (Gaussian Noise, Pitch Shift, Gain) are applied and processed using
    the Wav2Vec2FeatureExtractor.
    """
    def __init__(self, split: str = "train", sampling_rate: int = 16000, max_length: int = 3) -> None:
        """
        Initializes the WaveformDataset class.
        """
        super().__init__(split=split, subset_dir="AudioWaveformPreprocessor", ext=".pt")

        self.sampling_rate = sampling_rate
        self.max_length = max_length

        # Load the feature extractor from HuggingFace
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            BASE_MODEL,
            cache_dir=CACHE_DIR
        )

        # Define data augmentations pipeline (only applied during training)
        self.augmentations = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
            Gain(min_gain_db=-12, max_gain_db=12, p=0.5),
        ])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.files[idx]

        # Load tensor
        waveform = torch.load(file_path, weights_only=True)

        # Flatten if there are extra dimensions
        if waveform.dim() > 1:
            waveform = waveform.reshape(-1)

        # Apply augmentation
        if self.split == "train":
            waveform = self.augmentations(waveform, sample_rate=self.sampling_rate)

        # Process the waveform
        inputs = self.processor(
            waveform,
            sampling_rate=self.sampling_rate,
            padding="max_length",
            truncation=True,
            max_length=self.sampling_rate * self.max_length,
            return_tensors="pt",
            return_attention_mask=True
        )

        return {
            "input_values": inputs.input_values.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": torch.tensor(self._get_label(file_path), dtype=torch.long)
        }
