import os
import glob
import torch
from pathlib import Path
from torch.utils.data import Dataset


class RAVDESSEmotionDataset(Dataset):
    def __init__(self, data_path: str, split="train"):
        self.data_path = Path(data_path)
        self.split = split

        # Standard RAVDESS Mapping
        self.label2id = {
            "neutral": 0, "calm": 1, "happy": 2, "sad": 3,
            "angry": 4, "fearful": 5, "disgust": 6, "surprised": 7
        }

        self.emotion_map = {
            "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
        }

        # Load all data filtering everything out
        self.file_list = self._load_metadata()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # Load the waveform
        waveform = torch.load(file_path, weights_only=True)

        # 1. Handle stereo to mono: If [2, 48000], take the mean or first channel
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)

        # 2. Ensure it is strictly 1D [sequence_length]
        # This removes any remaining singleton dimensions like [1, 48000] -> [48000]
        waveform = waveform.flatten()

        parts = file_path.name.split('-')
        emotion_code = parts[2]

        label_name = self.emotion_map.get(emotion_code)
        label_id = self.label2id[label_name]

        return {
            "input_values": waveform,
            "labels": torch.tensor(label_id, dtype=torch.long)
        }

    def _load_metadata(self):
        # Use a more robust search pattern to find files deep in subdirectories
        search_pattern = os.path.join(self.data_path, "**", "*.pt")
        files = glob.glob(search_pattern, recursive=True)
        metadata = []

        train_actors = set(range(1, 21))
        val_actors = set(range(21, 23))
        test_actors = set(range(23, 25))

        for f in files:
            parts = os.path.basename(f).split('-')
            actor_id = int(parts[6].split('.')[0])  # The last part is actor ID
            emotion_code = parts[2]

            # FILTER: Skip files not in the current split
            if self.split == "train" and actor_id not in train_actors: continue
            if self.split == "val" and actor_id not in val_actors: continue
            if self.split == "test" and actor_id not in test_actors: continue

            if emotion_code in self.emotion_map:
                metadata.append(Path(f))
        return metadata


if __name__ == "__main__":
    dataset = RAVDESSEmotionDataset(data_path="../data/processed")
    dataset._load_metadata()

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Input Shape: {sample['input_values'].shape}")
        print(f"Label ID: {sample['labels'].item()}")
        print(f"Mapping: {dataset.label2id}")
    else:
        print("Please check if 'data/raw' contains 'Actor_XX' folders, if not download them using download_data.py")