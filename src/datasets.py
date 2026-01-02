import os
import glob
import torch
from pathlib import Path
from torch.utils.data import Dataset


class RAVDESSEmotionDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        # Recursively find all processed .pt files
        self.file_list = list(self.data_path.rglob("*.pt"))

        print(f"📊 Found {len(self.file_list)} processed files in {self.data_path}")

        # Standard RAVDESS Mapping
        self.label2id = {
            "neutral": 0, "calm": 1, "happy": 2, "sad": 3,
            "angry": 4, "fearful": 5, "disgust": 6, "surprised": 7
        }

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

        return {
            "input_values": waveform,
            "labels": torch.tensor(int(file_path.stem.split("-")[2]) - 1, dtype=torch.long)
        }

    def _load_metadata(self):
        # Use a more robust search pattern to find files deep in subdirectories
        search_pattern = os.path.join(self.data_path, "**", "*.wav")
        files = glob.glob(search_pattern, recursive=True)

        metadata = []
        for f in files:
            # RAVDESS format: 03-01-06-... (3rd element is emotion)
            parts = os.path.basename(f).split('-')
            if len(parts) >= 3:
                emotion_id = parts[2]
                if emotion_id in self.emotion_map:
                    metadata.append({
                        "path": f,
                        "label": self.label2id[self.emotion_map[emotion_id]]
                    })
        return metadata


if __name__ == "__main__":
    # Use 'data/raw' relative to where you run the script
    dataset = RAVDESSEmotionDataset(data_path="../data/raw")

    if len(dataset) > 0:
        print(f"✅ Success! Found {len(dataset)} audio files.")
        sample = dataset[0]
        print(f"Input Shape: {sample['input_values'].shape}")
        print(f"Label ID: {sample['labels'].item()}")
        print(f"Mapping: {dataset.label2id}")
    else:
        print("❌ Still found 0 files. Please check if 'data/raw' contains 'Actor_XX' folders.")