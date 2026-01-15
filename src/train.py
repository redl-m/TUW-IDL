#!/usr/bin/env python3

import numpy as np
import os
import torch
from sklearn.metrics import recall_score, accuracy_score
from transformers import Trainer, TrainingArguments

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))

from datasets import RAVDESSEmotionDataset
from model import create_ser_model

#TODO: implement additional model
# create graph of training loss and accuracy or table

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
    print(f"Best model saved to: {final_save_path}")

    metrics = trainer.evaluate()
    print(f"Final Results: Accuracy: {metrics['eval_accuracy']:.4f} | UAR: {metrics['eval_uar']:.4f}")

if __name__ == "__main__":
    train()