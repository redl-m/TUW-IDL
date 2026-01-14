from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from torch.utils.data import DataLoader

from datasets import RAVDESSEmotionDataset
from transformers import Wav2Vec2ForSequenceClassification


def evaluate_model():
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    model_path = root_dir / "models" / "best_ser_model_trainer"
    data_path = root_dir / "data" / "processed"
    results_path = root_dir / "results"
    results_path.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    dataset = RAVDESSEmotionDataset(data_path=str(data_path), split="test")

    if len(dataset) == 0:
        print(f"No files found in {data_path}")
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
    print(classification_report(all_labels, all_preds, target_names=list(dataset.label2id.keys())))

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=dataset.label2id.keys(), yticklabels=dataset.label2id.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Emotion Recognition Confusion Matrix')

    save_file = results_path / 'confusion_matrix.png'
    plt.savefig(save_file)
    print(f"Confusion matrix saved to {save_file}")

if __name__ == "__main__":
    evaluate_model()