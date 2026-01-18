#!/usr/bin/env python3

from preprocessors.base import AudioFilePreprocessor

from pathlib import Path
import numpy as np

import librosa
import librosa.feature as lf


class AudioFeatureExtractor(AudioFilePreprocessor):
    """
    Extracts features such as MFCC, Mel, Chroma and more from audio files.

    This class loads and cleans the audio, extracts statistical features,
    and final augments and normalizes the features.
    """
    def __init__(self, sample_rate: int = 22050) -> None:
        """
        Initializes the feature extractor.

        Note: this sampling rate (22050) is used as a default in librosa.
        """
        self.sample_rate = sample_rate

    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extracts a concatenated vector of features.
        """
        # Compute STFT (Short-Time Fourier Transform)
        stft = np.abs(librosa.stft(data))

        # Zero Crossing Rate
        zcr = np.mean(lf.zero_crossing_rate(y=data), axis=1)

        # Chroma (Pitch)
        chroma = np.mean(lf.chroma_stft(S=stft, sr=self.sample_rate), axis=1)

        # Mel Spectrogram (Energy)
        mel_spectrogram = lf.melspectrogram(S=stft ** 2, sr=self.sample_rate)
        mel_mean = np.mean(mel_spectrogram, axis=1)
        mel_std = np.std(mel_spectrogram, axis=1)
        mel_max = np.max(mel_spectrogram, axis=1)

        # MFCC (Timbre)
        mel_db = librosa.power_to_db(mel_spectrogram)
        mfcc = np.mean(lf.mfcc(S=mel_db, sr=self.sample_rate), axis=1)

        # RMS (Loudness)
        rms = np.mean(lf.rms(S=stft), axis=1)

        # Concatenate all into 1d array
        return np.concatenate([zcr, chroma, mfcc, rms, mel_mean, mel_max, mel_std])

    def _noise(self, data: np.ndarray) -> np.ndarray:
        """
        Applies random white noise to the audio.
        """
        noise_amp = 0.035 * np.random.uniform() * np.max(data)
        noise = noise_amp * np.random.normal(size=data.shape)
        return data + noise

    def _stretch(self, data: np.ndarray, rate: float = 0.8) -> np.ndarray:
        """
        Applies time stretching.
        """
        return librosa.effects.time_stretch(y=data, rate=rate)

    def _pitch(self, data: np.ndarray, n_steps: float = -2.0) -> np.ndarray:
        """
        Applies pitch shifting.
        """
        return librosa.effects.pitch_shift(y=data, sr=self.sample_rate, n_steps=n_steps)

    def process_file(self, input_path: Path, output_path: Path) -> None:
        # Load and clean audio
        try:
            data, _ = librosa.load(input_path, sr=self.sample_rate)

            # Trim leading/trailing silence
            data, _ = librosa.effects.trim(data, top_db=30)

            # Ensure length of 3s (pad or truncate)
            target_length = int(3 * self.sample_rate)
            data = librosa.util.fix_length(data, size=target_length)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            return

        # Extract features
        features_clean = self._extract_features(data)
        features_noise = self._extract_features(self._noise(data))

        # Stretch & Pitch Augmented
        data_aug = self._pitch(self._stretch(data, rate=0.8), n_steps=-2.0)
        features_aug = self._extract_features(data_aug)

        # Stack rows: Shape (3, N_Features)
        final_result = np.vstack([features_clean, features_noise, features_aug])

        # Apply normalization
        epsilon = 1e-7
        mean = final_result.mean(axis=1, keepdims=True)
        std = final_result.std(axis=1, keepdims=True)

        # Z-Score Normalization: (x - mean) / std
        final_result = (final_result - mean) / (std + epsilon)

        np.save(output_path.with_suffix(".npy"), final_result)