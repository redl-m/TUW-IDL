from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from datasets import RAVDESSEmotionDataset
from model import create_ser_model


def evaluate_model():
    # 0. Setup Absolute Paths
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    # Correct paths relative to the root
    model_path = root_dir / "models" / "ser_model.pt"
    # Note: It's better to evaluate on your PROCESSED data!
    data_path = root_dir / "data" / "processed"
    results_path = root_dir / "results"
    results_path.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data and Model using the new absolute paths
    dataset = RAVDESSEmotionDataset(data_path=str(data_path))

    if len(dataset) == 0:
        print(f"❌ Error: No files found in {data_path}")
        return

    model = create_ser_model(num_labels=8).to(device)
    model.load_state_dict(torch.load(str(model_path)))
    model.eval()

    all_preds = []
    all_labels = []

    # 2. Inference
    print("Evaluating on dataset...")
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            input_values = sample['input_values'].unsqueeze(0).to(device)
            label = sample['labels'].item()

            logits = model(input_values).logits
            pred = torch.argmax(logits, dim=1).item()

            all_preds.append(pred)
            all_labels.append(label)

    # 3. Calculate Project Metrics
    uar = recall_score(all_labels, all_preds, average='macro')
    print(f"\nUnweighted Average Recall (UAR): {uar:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(dataset.label2id.keys())))

    # 4. Qualitative Analysis: Confusion Matrix [cite: 19]
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=dataset.label2id.keys(), yticklabels=dataset.label2id.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Emotion Recognition Confusion Matrix')
    plt.savefig(results_path / 'confusion_matrix.png')
    print("Confusion matrix saved to results/confusion_matrix.png")


if __name__ == "__main__":
    evaluate_model()