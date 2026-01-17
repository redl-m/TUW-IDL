#!/usr/bin/env python3

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, accuracy_score

from models import CNN1DClassifier
from loaders import FeatureDataset
from utils.paths import RESULTS_DIR, MODELS_DIR, ensure_paths
from utils.config import LABEL2ID

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def train():
    model_path = MODELS_DIR / "best_features_model"
    feature_results_path = RESULTS_DIR / "feature"
    ensure_paths([feature_results_path, model_path])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    EPOCHS = 50

    try:
        train_ds = FeatureDataset(split="train")
        val_ds = FeatureDataset(split="val")
    except Exception as e:
        print(f"Error loading datasets. Ensure 'preprocess.py' was run. Details: {e}")
        return

    # weight calculations
    label_counts = Counter()
    for f in train_ds.files:
        l = train_ds._get_label(f)
        label_counts[l] += 1

    class_weights = []
    total_samples = sum(label_counts.values())
    num_classes = len(LABEL2ID)

    for i in range(num_classes):
        count = label_counts.get(i, 0)
        if count == 0:
            weight = 1.0
        else:
            weight = total_samples / (num_classes * count)
        class_weights.append(weight)

    weights_tensor = torch.tensor(class_weights).float().to(device)

    # Create standard PyTorch DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Detect dimensions
    sample_input = train_ds[0]["features"]
    input_dim = sample_input.shape[0]
    num_labels = 8

    model = CNN1DClassifier(input_dim=input_dim, num_labels=num_labels).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=False):
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        avg_train_loss = train_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]", leave=False):
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)

                # Get predictions
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_ds)

        # Calculate Metrics
        val_acc = accuracy_score(all_labels, all_preds)
        val_uar = recall_score(all_labels, all_preds, average='macro')

        # Store logs
        history.append({
            "epoch": epoch + 1,
            "loss": avg_train_loss,
            "eval_loss": avg_val_loss,
            "eval_accuracy": val_acc,
            "eval_uar": val_uar
        })

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_file = model_path / "model_weights.pth"
            torch.save(model.state_dict(), save_file)

    print(f"Best model with validation accuracy: {best_val_acc}")
    df_metrics = pd.DataFrame(history)

    csv_path = os.path.join(feature_results_path, "training_logs_features.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"Training logs saved to: {csv_path}")

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df_metrics, x='epoch', y='loss', label='Train Loss', marker='o')
    sns.lineplot(data=df_metrics, x='epoch', y='eval_loss', label='Validation Loss', marker='o')
    plt.title("Training vs Validation Loss (Features)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Plot Accuracy & Recall
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df_metrics, x='epoch', y='eval_accuracy', label='Val Accuracy', marker='s')
    sns.lineplot(data=df_metrics, x='epoch', y='eval_uar', label='Val UAR (Recall)', marker='s')
    plt.title("Validation Metrics (Features)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(feature_results_path, "training_history_features.png")
    plt.savefig(plot_path)
    print(f"Training plots saved to: {plot_path}")


if __name__ == "__main__":
    train()