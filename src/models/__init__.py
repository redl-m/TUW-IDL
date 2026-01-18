#!/usr/bin/env python3

"""
This module exposes the neural network architectures used for audio classification.
"""

from models.simple_classifier import SimpleAudioClassifier
from models.wav2vec2_model import Wav2Vec2Classifier

__all__ = ["SimpleAudioClassifier", "Wav2Vec2Classifier"]