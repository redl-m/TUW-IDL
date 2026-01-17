#!/usr/bin/env python3

from models.simple_classifier import SimpleAudioClassifier
from models.wav2vec2_model import create_wav2vec2_model

__all__ = ["SimpleAudioClassifier", "create_wav2vec2_model"]