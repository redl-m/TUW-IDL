#!/usr/bin/env python3

import os
import sys

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import recall_score, accuracy_score
from transformers import Trainer, TrainingArguments

from datasets import RAVDESSEmotionDataset
from model import create_ser_model

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    acc = accuracy_score(labels, preds)
    uar = recall_score(labels, preds, average='macro')

    return {"accuracy": acc, "uar": uar}


def train():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "processed")
    model_output_dir = os.path.join(project_root, "results")

    # Load datasets
    train_ds = RAVDESSEmotionDataset(data_path, split="train")
    val_ds = RAVDESSEmotionDataset(data_path, split="val")

    # Initialize model with 8 labels
    model = create_ser_model(num_labels=8)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
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
    final_save_path = os.path.join(project_root, "models", "best_ser_model_trainer")
    trainer.save_model(final_save_path)

    metrics = trainer.evaluate()
    print(f"Best model with accuracy of {metrics['eval_accuracy']:.4f} saved to: {final_save_path}")

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
    df_metrics.to_csv(os.path.join(model_output_dir, "training_logs.csv"), index=False)

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
    plot_path = os.path.join(model_output_dir, "training_history.png")
    plt.savefig(plot_path)
    print(f"\nTraining plots saved to: {plot_path}")

if __name__ == "__main__":
    train()