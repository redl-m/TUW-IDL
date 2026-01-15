#!/usr/bin/env python3

import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config

def create_ser_model(num_labels=8, model_name="facebook/wav2vec2-base"):
    """
    Builds the SER model using a pretrained wav2vec 2.0 encoder
    with a classification head.
    """
    # Load configuration and set label count
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task="audio-classification",
    )

    # Initialize model with pretrained weights
    # This includes the encoder + a linear layer (classification head)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )

    # Freeze the feature extractor to save memory and training time
    model.freeze_feature_extractor()

    return model

if __name__ == "__main__":
    model = create_ser_model(num_labels=8)

    # Print the model summary
    print("Model successfully initialized!")
    print(f"Model type: {type(model)}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")