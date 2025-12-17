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

# ==========================================
# 1. DATASET DEFINITION (RAVDESS)
# ==========================================
class RAVDESSEmotionDataset(Dataset):
    def __init__(self, data_path, processor_name="facebook/wav2vec2-base", max_duration=3.0):
        self.data_path = os.path.abspath(data_path)
        self.max_duration = max_duration
        self.sampling_rate = 16000 # Required for wav2vec 2.0 [cite: 6, 30]
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(processor_name)
        self.emotion_map = {'01':'neutral', '02':'calm', '03':'happy', '04':'sad',
                            '05':'angry', '06':'fearful', '07':'disgust', '08':'surprised'}
        self.label2id = {label: i for i, label in enumerate(self.emotion_map.values())}
        self.file_list = self._load_metadata()

    def _load_metadata(self):
        search_pattern = os.path.join(self.data_path, "**", "*.wav")
        files = glob.glob(search_pattern, recursive=True)
        metadata = []
        for f in files:
            parts = os.path.basename(f).split('-')
            if len(parts) >= 3:
                emotion_id = parts[2]
                if emotion_id in self.emotion_map:
                    metadata.append({"path": f, "label": self.label2id[self.emotion_map[emotion_id]]})
        return metadata

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        item = self.file_list[idx]
        speech, _ = librosa.load(item['path'], sr=self.sampling_rate)
        max_length = int(self.sampling_rate * self.max_duration)
        speech = librosa.util.fix_length(speech, size=max_length) # Standard preprocessing [cite: 13]
        inputs = self.processor(speech, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        return {"input_values": inputs.input_values.squeeze(0), "labels": torch.tensor(item['label'], dtype=torch.long)}

# ==========================================
# 2. MODEL DEFINITION (Option B: wav2vec 2.0)
# ==========================================
def create_ser_model(num_labels=8):
    # Fine-tuning a wav2vec 2.0 style encoder with a classification head [cite: 15, 34]
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels)
    return model

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training starting on: {device}")

    # Use the absolute path logic to find your data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "raw")

    full_dataset = RAVDESSEmotionDataset(data_path=data_path)
    if len(full_dataset) == 0:
        print(f"❌ Error: No files found at {data_path}")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True) # Small batch for CPU/lower RAM
    val_loader = DataLoader(val_ds, batch_size=4)

    model = create_ser_model(num_labels=8).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3): # Project suggests reporting metrics per epoch
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

    # Save for final evaluation scripts [cite: 25]
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/ser_model.pt")
    print("✅ Training complete! Model saved to models/ser_model.pt")

if __name__ == "__main__":
    train()