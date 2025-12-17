import os
import glob
import torch
import librosa
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor


class RAVDESSEmotionDataset(Dataset):
    def __init__(self, data_path, processor_name="facebook/wav2vec2-base", max_duration=3.0):
        # Convert to absolute path to avoid Windows location issues
        self.data_path = os.path.abspath(data_path)
        self.max_duration = max_duration
        self.sampling_rate = 16000

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(processor_name)

        self.emotion_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        self.label2id = {label: i for i, label in enumerate(self.emotion_map.values())}

        print(f"🔍 Searching for .wav files in: {self.data_path}")
        self.file_list = self._load_metadata()

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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if len(self.file_list) == 0:
            raise RuntimeError(f"No files found in {self.data_path}. Check your folder structure!")

        item = self.file_list[idx]
        speech, _ = librosa.load(item['path'], sr=self.sampling_rate)

        # Trimming/Padding as per project requirements
        max_length = int(self.sampling_rate * self.max_duration)
        if len(speech) > max_length:
            speech = speech[:max_length]
        else:
            speech = librosa.util.fix_length(speech, size=max_length)

        inputs = self.processor(speech, sampling_rate=self.sampling_rate, return_tensors="pt")

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(item['label'], dtype=torch.long)
        }


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