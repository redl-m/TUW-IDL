#!/usr/bin/env python3

import os
import glob
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor
from audiomentations import Compose, AddGaussianNoise, PitchShift, Gain

class RAVDESSEmotionDataset(Dataset):
    def __init__(self, data_path: str, split="train", sampling_rate=16000, max_length=3):
        self.data_path = Path(data_path)
        self.split = split
        self.sampling_rate = sampling_rate
        self.max_length = max_length

        # Initialize processor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')

        self.augmentations = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
            Gain(min_gain_db=-12, max_gain_db=12, p=0.5),
        ])

        self.label2id = {
            "neutral": 0, "calm": 1, "happy": 2, "sad": 3,
            "angry": 4, "fearful": 5, "disgust": 6, "surprised": 7
        }

        self.emotion_map = {
            "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
        }

        self.file_list = self._load_metadata()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # Load the pre-processed tensor
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

        input_values = inputs.input_values.squeeze()
        attention_mask = inputs.attention_mask.squeeze()

        # Extract emotion code from the filename
        parts = file_path.name.split('-')
        emotion_code = parts[2]

        label_name = self.emotion_map.get(emotion_code)
        label_id = self.label2id[label_name]

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_id, dtype=torch.long)
        }

    def _load_metadata(self):
        search_pattern = os.path.join(self.data_path, "**", "*.pt")
        files = glob.glob(search_pattern, recursive=True)
        metadata = []

        # Define actor IDs for strictly separated splits
        train_actors = set(range(1, 21))
        val_actors = set(range(21, 23))
        test_actors = set(range(23, 25))

        for f in files:
            parts = os.path.basename(f).split('-')
            actor_id = int(parts[6].split('.')[0])
            emotion_code = parts[2]

            # Skip files that do not belong to the current split
            if self.split == "train" and actor_id not in train_actors: continue
            if self.split == "val" and actor_id not in val_actors: continue
            if self.split == "test" and actor_id not in test_actors: continue

            if emotion_code in self.emotion_map:
                metadata.append(Path(f))

        return metadata


if __name__ == "__main__":
    # Quick sanity check
    dataset = RAVDESSEmotionDataset(data_path="../data/processed")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample loaded successfully.")
        print(f"Input shape: {sample['input_values'].shape}")
        print(f"Label ID: {sample['labels'].item()}")
    else:
        print("No data found. Check the processed folder.")