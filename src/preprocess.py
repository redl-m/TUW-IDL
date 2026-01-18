#!/usr/bin/env python3

"""
Script to run the preprocessing pipelines.
"""

from utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, ensure_paths
from preprocessors import AudioWaveformPreprocessor, AudioFeatureExtractor, AudioFilePreprocessor

import sys
from tqdm import tqdm


def run_preprocessing(preprocessors: list[AudioFilePreprocessor]) -> None:
    """
    Executes the processing logic for a list of preprocessors.
    """
    ensure_paths([DATA_RAW_DIR, DATA_PROCESSED_DIR])

    # Find all audiofiles
    audio_files = list(DATA_RAW_DIR.rglob("*.wav"))

    if not audio_files:
        print(f"No audio files found in {DATA_RAW_DIR}")
        return

    # Execute each preprocessor
    for preprocessor in preprocessors:
        preprocessor_name = preprocessor.__class__.__name__
        print(f"Executing {preprocessor_name} on {len(audio_files)} .wav files.")

        # Process all files
        for audio_path in tqdm(audio_files, desc=f"Processing"):
            # Determine output path
            relative_path = audio_path.relative_to(DATA_RAW_DIR)
            output_path = DATA_PROCESSED_DIR / preprocessor_name / relative_path

            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                preprocessor.process_file(input_path=audio_path, output_path=output_path)
            except Exception as e:
                print(f"Error processing {audio_path.name}: {e}")

if __name__ == "__main__":
    # Register CLI inputs to class
    model_registry = {
        "waveform": AudioWaveformPreprocessor,
        "features": AudioFeatureExtractor
    }

    selected_preprocessors = []
    args = sys.argv[1:]

    if not args:
        # Default to all preprocessors if no argument provided
        selected_preprocessors = [cls() for cls in model_registry.values()]
    else:
        for arg in args:
            if arg in model_registry:
                selected_preprocessors.append(model_registry[arg]())
            else:
                print(f"Unknown preprocessor: {arg}")

    if selected_preprocessors:
        run_preprocessing(selected_preprocessors)
    else:
        print("No preprocessors selected")