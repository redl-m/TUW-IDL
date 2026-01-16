#!/usr/bin/env python3

from utils.paths import DATA_RAW_DIR, DATA_PROCESSED_DIR, ensure_paths

from preprocessors.waveform import AudioWaveformPreprocessor
from preprocessors.features import AudioFeatureExtractor
from preprocessors.base import AudioFilePreprocessor

import sys
from tqdm import tqdm


def run_preprocessing(preprocessors: list[AudioFilePreprocessor]) -> None:
    ensure_paths([DATA_RAW_DIR, DATA_PROCESSED_DIR])
    audio_files = list(DATA_RAW_DIR.rglob("*.wav"))

    for preprocessor in preprocessors:
        preprocessor_name = preprocessor.__class__.__name__
        print(f"Processing {len(audio_files)} audio files with {preprocessor_name}")
        for audio_path in tqdm(audio_files, desc=f"Running {preprocessor_name}"):
            relative_path = audio_path.relative_to(DATA_RAW_DIR)
            output_path = DATA_PROCESSED_DIR / preprocessor_name / relative_path

            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                preprocessor.process_file(audio_path, output_path)
            except Exception as e:
                print(f"Error processing {audio_path.name}: {e}")

if __name__ == "__main__":
    model_registry = {
        "waveform": AudioWaveformPreprocessor,
        "features": AudioFeatureExtractor
    }

    selected_preprocessors = []
    args = sys.argv[1:]

    if not args:
        # All preprocessors if nothing specified
        selected_preprocessors = [cls() for cls in model_registry.values()]
    else:
        for arg in args:
            if arg in model_registry:
                selected_preprocessors.append(model_registry[arg])
            else:
                print(f"Unknown model: {arg}")

    if selected_preprocessors:
        run_preprocessing(selected_preprocessors)
    else:
        print("No preprocessors selected")