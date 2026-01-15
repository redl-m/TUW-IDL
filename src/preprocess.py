#!/usr/bin/env python3

import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm

class AudioPreprocessor:
    def __init__(self, target_sr=16000, duration_sec=3.0):
        self.target_sr = target_sr
        self.num_samples = int(target_sr * duration_sec)

    def process_file(self, input_path, output_path):
        waveform, sr = torchaudio.load(input_path)

        # Resample 48kHz to 16kHz if necessary
        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)

        # Standardize length (3 seconds = 48,000 samples at 16kHz)
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        else:
            padding = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Save the raw waveform tensor with shape [1, 48000]
        torch.save(waveform, output_path.with_suffix(".pt"))

def run_preprocessing():
    root = Path(__file__).parent.parent
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"

    preprocessor = AudioPreprocessor()
    audio_files = list(raw_dir.rglob("*.wav"))

    print(f"Found {len(audio_files)} audio files. Starting preprocessing.")

    for audio_path in tqdm(audio_files):
        relative_path = audio_path.relative_to(raw_dir)
        output_path = processed_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            preprocessor.process_file(audio_path, output_path)
        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")

if __name__ == "__main__":
    run_preprocessing()