import os
import glob
import torch
import torch.nn as nn
import librosa
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from sklearn.metrics import recall_score, accuracy_score, f1_score
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Config

script_dir = os.path.dirname(os.path.abspath(__file__))

from datasets import RAVDESSEmotionDataset
from model import create_ser_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training starting on: {device} ({torch.cuda.get_device_name(0)})")

    # Use the absolute path logic to find your data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "processed")

    train_ds = RAVDESSEmotionDataset(data_path, split="train")
    val_ds = RAVDESSEmotionDataset(data_path, split="val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True) # Small batch for CPU/lower RAM
    val_loader = DataLoader(val_ds, batch_size=4)

    model = create_ser_model(num_labels=8).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Save for final evaluation scripts
    save_dir = os.path.join(script_dir, "..", "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ser_model.pt")

    best_uar = 0.0

    for epoch in range(5): # Project suggests reporting metrics per epoch
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs, labels = batch['input_values'].to(device), batch['labels'].to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['input_values'].to(device)).logits
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].numpy())

        # Report Required Metrics
        uar = recall_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        print(f"📊 Epoch {epoch+1} Results -> Accuracy: {acc:.4f}, UAR: {uar:.4f}")

        if uar > best_uar:
            best_uar = uar
            torch.save(model.state_dict(), save_path)
            print(f"   🔥 New Best Model Found! Saved to {os.path.basename(save_path)}")

    print(f"✅ Training complete! Best UAR achieved: {best_uar:.4f}")

if __name__ == "__main__":
    train()