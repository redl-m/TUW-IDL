#!/usr/bin/env python3

from models.simple_classifier import CNN1DClassifier
from models.wav2vec2_model import create_wav2vec2_model

__all__ = ["CNN1DClassifier", "create_wav2vec2_model"]