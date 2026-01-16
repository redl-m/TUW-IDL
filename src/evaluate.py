#!/usr/bin/env python3

from loaders import WaveformDataset
from utils.paths import RESULTS_DIR, MODELS_DIR, DATA_PROCESSED_DIR, ensure_paths
from utils.config import LABEL2ID

import matplotlib.pyplot as plt

import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification


def evaluate_model():
    model_path = MODELS_DIR / "best_waveform_model"
    ensure_paths([RESULTS_DIR, model_path])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    dataset = WaveformDataset(split="test")

    if len(dataset) == 0:
        print(f"No files found in {DATA_PROCESSED_DIR}")
        return

    # Use a DataLoader for batch processing (faster than single items)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Load the trained model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    ).to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating on dataset...")
    with torch.no_grad():
        for batch in test_loader:
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_values).logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    uar = recall_score(all_labels, all_preds, average='macro')
    print(f"\nUnweighted Average Recall (UAR): {uar:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(LABEL2ID.keys())))

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=LABEL2ID.keys(), yticklabels=LABEL2ID.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Emotion Recognition Confusion Matrix')

    save_file = RESULTS_DIR / 'confusion_matrix.png'
    plt.savefig(save_file)
    print(f"Confusion matrix saved to {save_file}")

if __name__ == "__main__":
    evaluate_model()