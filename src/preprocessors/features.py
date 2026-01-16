#!/usr/bin/env python3

from preprocessors.base import AudioFilePreprocessor

from pathlib import Path
import numpy as np

import librosa
import librosa.feature as lf


class AudioFeatureExtractor(AudioFilePreprocessor):
    def __init__(self, sample_rate: int = 22050) -> None:
        """
        Initializes the feature extractor and sets up values used later on.
        """
        self.sample_rate = sample_rate

    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extracts features efficiently by reusing the Spectrogram.
        """
        # Compute STFT
        stft = np.abs(librosa.stft(data))

        # Zero Crossing Rate
        zcr = np.mean(lf.zero_crossing_rate(y=data), axis=1)

        # Chroma
        chroma = np.mean(lf.chroma_stft(S=stft, sr=self.sample_rate), axis=1)

        # Mel Spectrogram
        mel_spectrogram = lf.melspectrogram(S=stft ** 2, sr=self.sample_rate)
        mel_mean = np.mean(mel_spectrogram, axis=1)
        mel_std = np.std(mel_spectrogram, axis=1)
        mel_max = np.max(mel_spectrogram, axis=1)

        # MFCC
        mel_db = librosa.power_to_db(mel_spectrogram)
        mfcc = np.mean(lf.mfcc(S=mel_db, sr=self.sample_rate), axis=1)

        # RMS
        rms = np.mean(lf.rms(S=stft), axis=1)

        # Concatenate all
        return np.concatenate([zcr, chroma, mfcc, rms, mel_mean, mel_max, mel_std])

    def _noise(self, data: np.ndarray) -> np.ndarray:
        """
        Add random white noice
        """
        noise_amp = 0.035 * np.random.uniform() * np.max(data)
        noise = noise_amp * np.random.normal(size=data.shape)
        return data + noise

    def _stretch(self, data: np.ndarray, rate: float = 0.8) -> np.ndarray:
        """
        Time stretching.
        """
        return librosa.effects.time_stretch(y=data, rate=rate)

    def _pitch(self, data: np.ndarray, n_steps: float = -2.0) -> np.ndarray:
        """
        Pitch shifting.
        """
        return librosa.effects.pitch_shift(y=data, sr=self.sample_rate, n_steps=n_steps)

    def process_file(self, input_path: Path, output_path: Path) -> None:
        # Load audio (only once)
        try:
            data, _ = librosa.load(input_path, sr=self.sample_rate, duration=2.5, offset=0.6)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            return

        features_clean = self._extract_features(data)
        features_noise = self._extract_features(self._noise(data))

        # Stretch & Pitch Augmented
        data_aug = self._pitch(self._stretch(data, rate=0.8), n_steps=-2.0)
        features_aug = self._extract_features(data_aug)

        # Stack rows: Shape (3, N_Features)
        final_result = np.vstack([features_clean, features_noise, features_aug])
        np.save(output_path.with_suffix(".npy"), final_result)