#!/usr/bin/env python3

import os
import sys

from loaders import WaveformDataset
from models import create_wav2vec2_model
from utils.paths import RESULTS_DIR, MODELS_DIR, ensure_paths

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import recall_score, accuracy_score
from transformers import Trainer, TrainingArguments

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    acc = accuracy_score(labels, preds)
    uar = recall_score(labels, preds, average='macro')

    return {"accuracy": acc, "uar": uar}

def train():
    model_path = MODELS_DIR / "best_waveform_model"
    waveform_results_path = RESULTS_DIR / "waveform"
    ensure_paths([waveform_results_path, model_path])

    # Load datasets
    train_ds = WaveformDataset(split="train")
    val_ds = WaveformDataset(split="val")

    # Initialize model with 8 labels
    model = create_wav2vec2_model(num_labels=8)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=waveform_results_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=30,
        weight_decay=0.01,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False
    )

    # Set up the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    print("Starting training with Hugging Face Trainer...")
    trainer.train()

    # Save the final model
    trainer.save_model(model_path)

    metrics = trainer.evaluate()
    print(f"Best model with accuracy of {metrics['eval_accuracy']:.4f} saved to: {model_path}")

    history = trainer.state.log_history

    train_logs = [x for x in history if 'loss' in x and 'epoch' in x]
    eval_logs = [x for x in history if 'eval_loss' in x and 'epoch' in x]

    df_train = pd.DataFrame(train_logs)
    df_eval = pd.DataFrame(eval_logs)

    df_train_epoch = df_train.groupby('epoch')['loss'].mean().reset_index()
    df_metrics = pd.merge(df_train_epoch, df_eval[['epoch', 'eval_loss', 'eval_accuracy', 'eval_uar']],
                          on='epoch', how='inner')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Save metrics to CSV
    df_metrics.to_csv(os.path.join(waveform_results_path, "training_logs_waveform.csv"), index=False)

    # Plot the Results
    plt.figure(figsize=(12, 5))

    # Plot - Loss Curve
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df_metrics, x='epoch', y='loss', label='Train Loss', marker='o')
    sns.lineplot(data=df_metrics, x='epoch', y='eval_loss', label='Validation Loss', marker='o')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Plot - Accuracy & UAR
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df_metrics, x='epoch', y='eval_accuracy', label='Val Accuracy', marker='s')
    sns.lineplot(data=df_metrics, x='epoch', y='eval_uar', label='Val UAR (Recall)', marker='s')
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(waveform_results_path, "training_history_waveform.png")
    plt.savefig(plot_path)
    print(f"\nTraining plots saved to: {plot_path}")

if __name__ == "__main__":
    train()