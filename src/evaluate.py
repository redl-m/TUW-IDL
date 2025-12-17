import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from dataset import RAVDESSEmotionDataset
from model import create_ser_model


def evaluate_model(model_path="models/ser_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data and Model
    dataset = RAVDESSEmotionDataset(data_path="data/raw")
    model = create_ser_model(num_labels=8).to(device)
    model.load_state_dict(torch.load(model_path))
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
    plt.savefig('results/confusion_matrix.png')
    print("Confusion matrix saved to results/confusion_matrix.png")


if __name__ == "__main__":
    evaluate_model()