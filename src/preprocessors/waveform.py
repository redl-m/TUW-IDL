#!/usr/bin/env python3

from preprocessors.base import AudioFilePreprocessor

import torch
import torchaudio
import torchaudio.transforms as T


class AudioWaveformPreprocessor(AudioFilePreprocessor):
    """
    Preprocesses raw audio files into fixed length tensors. It handles resampling
    to 16kHz, converting stereo to mono, and padding/truncating.
    """
    def __init__(self, target_sr: int = 16000, duration_sec: float = 3.0) -> None:
        """
        Initializes the waveform preprocessor.
        """
        self.target_sr = target_sr
        self.num_samples = int(target_sr * duration_sec)

    def process_file(self, input_path, output_path):
        waveform, sr = torchaudio.load(input_path.absolute())

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