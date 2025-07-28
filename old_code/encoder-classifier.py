import os
import time
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.transforms import MelSpectrogram
from torchvggish import vggish, vggish_input

from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

from pathlib import Path
from dataclasses import dataclass

from transformers import WhisperProcessor, WhisperModel


# ------------------ Config ------------------
SAMPLE_RATE = 16000
SEGMENT_LENGTH = 10 * SAMPLE_RATE
mel_kwargs = {
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 64,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
BREATHING_LABELS = {
    "breathing": 0,           # Normal breathing
    "light breathing": 1,     # Light breathing
    "heavy breathing": 2,     # Heavy breathing
    "heavy breathing with noises": 3,  # Heavy breathing with noises
    "no breathing": 4         # No breathing
}

# ------------------ New Training Params ------------------
@dataclass
class TrainingParams:
    num_epochs: int = 1000
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 100
    eval_every: int = 10
    train_csv_path: str = "train_log.csv"
    val_csv_path: str = "val_log.csv"
    best_model_path: str = "best_model.pt"

# ------------------ Utils ------------------
def load_labels(label_path):
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            onset, offset, label = line.strip().split(maxsplit=2)
            labels.append((float(onset), float(offset), label.lower().strip()))
    return labels

def is_breathing_segment(start, end, label_entries):
    for onset, offset, label in label_entries:
        label = label.lower().strip()
        # If overlap exists between the segment and the label time
        if max(onset, start) < min(offset, end):
            # Find the most specific (longest) matching breathing type
            best_match = None
            best_match_length = 0
            best_class_idx = None
            
            for breathing_type, class_idx in BREATHING_LABELS.items():
                if breathing_type in label and len(breathing_type) > best_match_length:
                    best_match = breathing_type
                    best_match_length = len(breathing_type)
                    best_class_idx = class_idx
            
            if best_match is not None:
                return best_class_idx
    
    return 4  # Default to "no breathing" if no overlap found


# ------------------ Dataset ------------------
class BreathingSegmentDataset(Dataset):
    def __init__(self, audio_dir, label_dir, segment_length=SEGMENT_LENGTH, transform=None):
        self.data = []
        self.transform = transform
        audio_files = glob.glob(f"{audio_dir}/**/*.wav", recursive=True)
        for audio_path in audio_files:
            file_id = Path(audio_path).stem
            label_path = os.path.join(label_dir, f"{file_id}.txt")
            if not os.path.exists(label_path):
                print(f"Label file not found for {audio_path}, skipping...")
                continue
            labels = load_labels(label_path)
            waveform, sr = torchaudio.load(audio_path)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

            for start in range(0, waveform.shape[1] - segment_length, segment_length):
                segment = waveform[:, start:start+segment_length]
                label = is_breathing_segment(start/SAMPLE_RATE, (start+segment_length)/SAMPLE_RATE, labels)
                self.data.append((segment, label))

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     waveform, label = self.data[idx]
    #     if self.transform:
    #         features = self.transform(waveform)
    #     else:
    #         features = waveform
    #     return features.squeeze(0), label
    
    def __getitem__(self, idx):
        waveform, label = self.data[idx]
        return waveform, label  # keep raw waveform

class SMOTEDataset(Dataset):
    def __init__(self, dataset, random_state=42):
        """
        Initialize the dataset with original dataset and perform SMOTE resampling internally.
        
        Args:
            dataset: The original dataset to apply SMOTE to
            random_state: Random seed for reproducibility
        """
        self.original_dataset = dataset
        self.random_state = random_state
        self.data = []
        self.labels = []
        
        # Extract data from original dataset
        X_original = []
        y_original = []
        
        print("Extracting data from original dataset...")
        for i in range(len(self.original_dataset)):
            waveform, label = self.original_dataset[i]
            # Store the original shape for the first item
            if i == 0:
                self.original_shape = waveform.shape
            
            # Convert waveform tensor to flattened numpy array for SMOTE
            X_original.append(waveform.flatten().numpy())
            y_original.append(label)
        
        X_original = np.array(X_original)
        y_original = np.array(y_original)
        
        # Check class distribution before SMOTE
        print("Original class distribution:", Counter(y_original))
        
        # Apply SMOTE
        print("Applying SMOTE resampling...")
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_original, y_original)
        
        # Check class distribution after SMOTE
        print("Resampled class distribution:", Counter(y_resampled))
        
        # Process the resampled data
        self._preprocess(X_resampled, y_resampled)
        
        print(f"SMOTE preprocessing complete. Dataset size: {len(self.data)} samples")

    def _preprocess(self, X, y):
        """
        Preprocess the resampled data by reshaping and converting to torch tensors.
        
        Args:
            X: Resampled feature data (flattened waveforms)
            y: Corresponding labels
        """
        for i in range(len(y)):
            waveform = torch.tensor(X[i].reshape(self.original_shape), dtype=torch.float32)
            label = int(y[i])
            self.data.append(waveform)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ------------------ Encoder ------------------
class VGGishEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.vggish = vggish(postprocess=False)
        self.vggish.eval()  # freeze pretrained layers
        self.proj = nn.Linear(128, out_dim)  # VGGish outputs 128-dim features

    def forward(self, x):
        # x shape: [B, 1, T], where T is audio length
        embeddings = []
        for i in range(x.size(0)):
            waveform = x[i].squeeze().cpu().numpy()  # [T]
            examples = vggish_input.waveform_to_examples(waveform, SAMPLE_RATE)  # [N, 96, 64]

            examples = examples.float().to(x.device)  # [N, 1, 96, 64]
            with torch.no_grad():
                feat = self.vggish(examples)  # [N, 128]
            embeddings.append(feat.mean(dim=0))  # average over time windows

        embeddings = torch.stack(embeddings).to(x.device)  # [B, 128]
        return self.proj(embeddings)  # [B, out_dim]

class WhisperEncoder(nn.Module):
    def __init__(self, model_name="openai/whisper-small", out_dim=64):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name)
        self.model.eval()  # Freeze the encoder
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Projection layer to match desired output dim
        self.proj = nn.Linear(self.model.config.d_model, out_dim)

    def forward(self, x):
        # x: [B, 1, T] in waveform format
        embeddings = []
        for i in range(x.size(0)):
            waveform = x[i].squeeze().cpu()
            input_features = self.processor.feature_extractor(
                waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features  # shape: [1, 80, 3000]

            with torch.no_grad():
                encoder_outputs = self.model.encoder(input_features.to(x.device))
                # Mean-pooling across time dimension
                emb = encoder_outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]
                embeddings.append(emb.squeeze(0))

        embeddings = torch.stack(embeddings)  # [B, hidden_dim]
        return self.proj(embeddings)  # [B, out_dim]


# ------------------ Modified Encoder ------------------
class VGGishClassifier(nn.Module):
    def __init__(self, out_dim=64, num_classes=2):
        super().__init__()
        self.encoder = VGGishEncoder(out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)
        
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

# ------------------ New Training Loop ------------------
def train_model(train_loader, val_loader, model, params: TrainingParams):
    start_time = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=params.learning_rate,
                               weight_decay=params.weight_decay)
    
    best_loss = float('inf')  # Initialize to infinity for minimization
    best_acc = 0.0
    no_improve = 0

    # Lists to collect metrics
    train_metrics = []
    val_metrics = []
    
    for epoch in range(params.num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = total_loss / total
        train_acc = 100 * correct / total
        
        train_metrics.append({'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc})

        # Only evaluate every n epochs
        if (epoch + 1) % params.eval_every == 0 or epoch == params.num_epochs - 1:
            model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Evaluation"):
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    total_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = 100 * correct / total
            val_loss = total_loss / total
            print(f"Epoch {epoch+1}/{params.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            
            # Append metrics for this epoch
            val_metrics.append({'epoch': epoch+1, 'val_loss': val_loss, 'val_acc': val_acc})    

            if val_loss < best_loss:
                best_loss = val_loss
                best_acc = val_acc
                no_improve = 0
                print(f"New best model found at epoch {epoch+1} with val acc: {val_acc:.2f}% and loss: {val_loss:.4f}")
                torch.save(model.state_dict(), "best_model.pt")
            else:
                no_improve += 1

            if no_improve >= params.patience:
                print("Early stopping")
                break

    # Save metrics to CSV using pandas
    pd.DataFrame(train_metrics).to_csv(params.train_csv_path, index=False)
    pd.DataFrame(val_metrics).to_csv(params.val_csv_path, index=False)
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Best model saved to {params.best_model_path} with val acc {best_acc:.4f}, val loss {best_loss:.4f}")
    with open("train_log.txt", "w") as f:
        f.write(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
        f.write(f"Best validation accuracy: {best_acc:.2f}% at epoch {epoch+1}\n")
        f.write(f"Best validation loss: {best_loss:.4f}\n")

# ------------------ Modified Evaluation ------------------
def evaluate_model(loader, model, criterion, log_path="test_eval_log.csv", cm_path="test_confusion_matrix.csv"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Evaluation Progress"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    acc = 100 * correct / total
    avg_loss = total_loss / total

    # === Metrics ===
    print(f"\nFinal Evaluation on Test Set:")
    print(f"Accuracy: {acc:.2f}%, Loss: {avg_loss:.4f}")

    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, output_dict=True)
    verbose_report = classification_report(all_labels, all_preds)
    print(verbose_report)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    log_data = {
        "test_accuracy": [acc],
        "test_loss": [avg_loss],
    }
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                log_data[f"{label}_{metric}"] = [value]
    pd.DataFrame(log_data).to_csv(log_path, index=False)
    print(f"Test evaluation log saved to {log_path}")

    # === Save Confusion Matrix ===
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(cm_path, index=False)
    print(f"Confusion matrix saved to {cm_path}")

    # === Save Classification Report ===
    results_path = "result.txt"
    with open(results_path, "w") as f:
        f.write(f"Final Evaluation on Test Set:\n")
        f.write(f"Accuracy: {acc:.2f}%, Loss: {avg_loss:.4f}\n\n")
        f.write("=== Classification Report ===\n")
        f.write(str(verbose_report))
    print(f"Classification report saved to {results_path}")

# ------------------ Updated Main Execution ------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    # Dataset and DataLoader setup
    transform = MelSpectrogram(sample_rate=SAMPLE_RATE, **mel_kwargs)
    dataset = BreathingSegmentDataset("audio", "strong_labels", transform=transform)
    
    # Split dataset
    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
   
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Training set size: {len(train_set)} samples")
    print(f"Validation set size: {len(val_set)} samples")
    print(f"Test set size: {len(test_set)} samples")
    
    # Print class distribution
    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("Class distribution:")
    for class_idx, count in class_counts.items():
        class_name = [k for k, v in BREATHING_LABELS.items() if v == class_idx][0]
        print(f"  {class_name}: {count} samples")

    # Apply SMOTE to the training set
    train_set_resampled = SMOTEDataset(train_set)

    # Create DataLoaders
    params = TrainingParams()
    train_loader = DataLoader(train_set_resampled, batch_size=params.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params.batch_size)
    test_loader = DataLoader(test_set, batch_size=params.batch_size)
    
    # Initialize model
    model = VGGishClassifier(out_dim=64, num_classes=5).to(DEVICE)
    
    # Train
    train_model(train_loader, val_loader, model, params)
    
    # Final evaluation
    model.load_state_dict(torch.load(params.best_model_path, weights_only=True))
    evaluate_model(test_loader, model, nn.CrossEntropyLoss())
