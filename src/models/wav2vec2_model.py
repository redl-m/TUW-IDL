#!/usr/bin/env python3

from utils.config import ID2LABEL, LABEL2ID, BASE_MODEL, CACHE_DIR, NUM_LABELS

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config


class Wav2Vec2Classifier(nn.Module):
    """
    A wrapper class for the HuggingFace Wav2Vec2 model for Sequence Classification.
    """
    def __init__(
            self,
            model_name: str = BASE_MODEL,
            num_labels: int = NUM_LABELS,
            label2id: Dict[str, int] = LABEL2ID,
            id2label: Dict[int, str] = ID2LABEL
    ) -> None:
        """
        Initializes the Wav2Vec2 classifier.
        """
        super().__init__()

        # Configure the model Architecture
        self.config = Wav2Vec2Config.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            finetuning_task="emotion_recognition",
            cache_dir=CACHE_DIR
        )

        # Load the pre-trained Weights with the classification head
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            ignore_mismatched_sizes=True,
            cache_dir=CACHE_DIR
        )

        # Freeze feature encoder to speed up training
        self.wav2vec2.freeze_feature_encoder()

    def forward(
            self,
            input_values: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass of the model.
        """
        return self.wav2vec2(
            input_values=input_values,
            labels=labels,
            attention_mask=attention_mask
        )

    def __getattr__(self, name: str):
        """
        Forward any unknown attribute access directly to the internal Hugging Face model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wav2vec2, name)

    def save_pretrained(self, save_directory: str) -> None:
        """
        Allows the Trainer to save the internal model directly using standard HF formats.
        """
        self.wav2vec2.save_pretrained(save_directory)

if __name__ == "__main__":
    model = Wav2Vec2Classifier()

    # Print the model summary
    print("Model successfully initialized!")
    print(f"Model type: {type(model)}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")