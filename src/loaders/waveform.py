#!/usr/bin/env python3

from loaders.base import BaseRAVDESSDataset
from utils.paths import CACHE_DIR

import torch
from transformers import Wav2Vec2FeatureExtractor
from audiomentations import Compose, AddGaussianNoise, PitchShift, Gain


class WaveformDataset(BaseRAVDESSDataset):
    def __init__(self, split="train", sampling_rate=16000, max_length=3):
        super().__init__(split=split, subset_dir="AudioWaveformPreprocessor", ext=".pt")

        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            'facebook/wav2vec2-base',
            cache_dir=CACHE_DIR
        )

        self.augmentations = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
            Gain(min_gain_db=-12, max_gain_db=12, p=0.5),
        ])

    def __getitem__(self, idx):
        file_path = self.files[idx]
        waveform = torch.load(file_path, weights_only=True)

        # Flatten if there are extra dimensions
        if waveform.dim() > 1:
            waveform = waveform.reshape(-1)

        # Apply augmentation
        if self.split == "train":
            waveform = self.augmentations(waveform, sample_rate=self.sampling_rate)

        # Use the processor for normalization
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
