#!/usr/bin/env python3

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config
from utils.config import ID2LABEL, LABEL2ID
from utils.paths import CACHE_DIR

def create_wav2vec2_model(
    model_name: str = "facebook/wav2vec2-base",
    num_labels: int = 8,
    label2id: dict = LABEL2ID,
    id2label: dict = ID2LABEL
):
    """
    Initializes a Wav2Vec2 model for Sequence Classification.
    """

    # Configure the model architecture
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        finetuning_task="emotion_recognition",
        cache_dir=CACHE_DIR
    )

    # Load the pre-trained weights with the specific classification head
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,  # Necessary if resizing the classification head
        cache_dir=CACHE_DIR
    )

    # Freeze feature encoder to speed up training (optional but recommended)
    model.freeze_feature_encoder()

    return model

if __name__ == "__main__":
    model = create_wav2vec2_model()

    # Print the model summary
    print("Model successfully initialized!")
    print(f"Model type: {type(model)}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")