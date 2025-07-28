import os
import time
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.transforms import MelSpectrogram
from torchvggish import vggish, vggish_input

from sklearn.metrics import classification_report, confusion_matrix

from pathlib import Path
from dataclasses import dataclass

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
SEGMENT_LENGTH = 16000  # 1 second at 16kHz
BREATHING_LABELS = {"breathing", "clean breathing", "heavy breathing"}

# Mel spectrogram parameters
mel_kwargs = {
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 64,
    "f_min": 20,
    "f_max": 8000,
}

class TrainingParams:
    def __init__(
        self,
        n_episodes=30000,
        n_way=2,
        k_shot=5,
        q_query=5,
        learning_rate=0.001,
        eval_every=1000,
        save_train_log=None,
        save_eval_log=None,
        best_model_path=None,
    ):
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.learning_rate = learning_rate
        self.eval_every = eval_every
        self.save_train_log = save_train_log
        self.save_eval_log = save_eval_log
        self.best_model_path = best_model_path

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
        if label in BREATHING_LABELS or "breathing" in label:
            # If overlap exists between the segment and the label time
            if max(onset, start) < min(offset, end):
                return 1
    return 0

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


class QuantumInspiredEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, out_dim=64, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, **mel_kwargs)
        
        # Calculate the flattened dimension first
        flatten_dim = self._calculate_flatten_dim()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(flatten_dim, hidden_dim)
        )
        
        # Quantum-inspired unitary transforms
        self.unitary_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) 
            for _ in range(n_layers)
        ])
        
        # Initialize unitary layers with orthogonal matrices
        for layer in self.unitary_layers:
            nn.init.orthogonal_(layer.weight)
        
        # Phase shift activation (complex-valued behavior)
        self.phase_shifts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()  # Maps to [-1, 1] like phase
            ) for _ in range(n_layers)
        ])
        
        # Final projection
        self.projection = nn.Linear(hidden_dim, out_dim)
    
    def _calculate_flatten_dim(self):
        # Calculate the flattened dimension after convolutions
        with torch.no_grad():
            dummy_waveform = torch.zeros(1, SEGMENT_LENGTH)
            mel_spec = self.mel_transform(dummy_waveform)
            mel_spec = mel_spec.unsqueeze(0)  # Add batch and channel dims: [1, 1, n_mels, time]
            
            # Manually apply the convolution operations
            x = mel_spec
            # First conv block
            x = nn.Conv2d(1, 32, 3, padding=1)(x)
            x = nn.ReLU()(x)
            x = nn.MaxPool2d(2)(x)
            # Second conv block
            x = nn.Conv2d(32, 64, 3, padding=1)(x)
            x = nn.ReLU()(x)
            x = nn.MaxPool2d(2)(x)
            # Get flattened shape
            return x.view(1, -1).shape[1]
    
    def forward(self, x):
        # x is raw waveform: [B, 1, T]
        batch_size = x.size(0)
        
        # Convert to mel spectrogram
        specs = []
        for i in range(batch_size):
            mel_spec = self.mel_transform(x[i].squeeze(0))  # [n_mels, time]
            specs.append(mel_spec.unsqueeze(0))  # Add channel dim: [1, n_mels, time]
        
        x = torch.stack(specs)  # [B, 1, n_mels, time]
        
        # Extract classical features
        x = self.feature_extractor(x)  # [B, hidden_dim]
        
        # Apply quantum-inspired unitary transforms and phase shifts
        for i in range(self.n_layers):
            # Apply unitary transformation (preserves norm)
            unitary = self.unitary_layers[i]
            phase = self.phase_shifts[i](x)
            
            # Complex multiplication simulation: 
            # Apply unitary transform then modulate with phase
            x = unitary(x) * (1 + phase)
            
            # Apply non-linearity that preserves some quantum properties
            x = nn.functional.normalize(x, p=2, dim=1) * torch.sqrt(torch.tensor(self.hidden_dim).float())
        
        # Project to output dimension
        x = self.projection(x)
        
        return x


def create_episode(dataset, n_way=5, k_shot=5, q_query=5):
    """Create a few-shot episode from dataset."""
    # Get unique classes
    labels = set()
    for _, label in dataset:
        labels.add(label)
    labels = list(labels)
    
    # Randomly select n_way classes
    episode_classes = random.sample(labels, n_way)
    
    support_x = []
    support_y = []
    query_x = []
    query_y = []
    
    # For each class, select k_shot + q_query examples
    for class_idx, class_label in enumerate(episode_classes):
        # Find all examples of this class
        class_examples = [(i, x) for i, (x, y) in enumerate(dataset) if y == class_label]
        
        # Randomly select k_shot + q_query examples
        selected = random.sample(class_examples, k_shot + q_query)
        
        # Split into support and query
        for i, (sample_idx, x) in enumerate(selected):
            if i < k_shot:
                support_x.append(x)
                support_y.append(class_idx)
            else:
                query_x.append(x)
                query_y.append(class_idx)
    
    # Convert to tensors
    support_x = torch.stack(support_x).unsqueeze(1).to(DEVICE)  # [n_way*k_shot, 1, T]
    support_y = torch.tensor(support_y).to(DEVICE)
    query_x = torch.stack(query_x).unsqueeze(1).to(DEVICE)  # [n_way*q_query, 1, T]
    query_y = torch.tensor(query_y).to(DEVICE)
    
    return support_x, support_y, query_x, query_y

def compute_prototypes(embeddings, labels):
    """Compute class prototypes from support embeddings."""
    classes = torch.unique(labels)
    prototypes = []
    proto_labels = []
    
    for c in classes:
        # Select embeddings of class c
        class_mask = (labels == c)
        class_embeddings = embeddings[class_mask]
        
        # Compute prototype as mean of embeddings
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)
        proto_labels.append(c.item())
    
    return torch.stack(prototypes), torch.tensor(proto_labels)

def quantum_inspired_prototypical_loss(encoder, support, support_labels, query, query_labels):
    support_embeddings = encoder(support)
    query_embeddings = encoder(query)

    prototypes, proto_labels = compute_prototypes(support_embeddings, support_labels)
    
    # Normalize embeddings to unit sphere (quantum state-like)
    query_norm = F.normalize(query_embeddings, p=2, dim=1)
    proto_norm = F.normalize(prototypes, p=2, dim=1)
    
    # Calculate fidelity-inspired similarity (quantum-inspired)
    # Using squared inner product as in quantum fidelity
    inner_products = torch.mm(query_norm, proto_norm.t())
    fidelities = inner_products.pow(2)
    
    # Convert to log probabilities
    log_p_y = F.log_softmax(fidelities * 10.0, dim=1)  # Scale factor for sharper distributions
    
    target_inds = torch.tensor([torch.where(proto_labels == y)[0].item() for y in query_labels])
    loss = F.nll_loss(-log_p_y, target_inds.to(DEVICE))
    acc = (log_p_y.argmax(dim=1) == target_inds.to(DEVICE)).float().mean()
    
    return loss, acc

def train_quantum_protonet_with_eval(train_set, val_set, encoder, params: TrainingParams):
    start_time = time.time()

    optimizer = torch.optim.Adam(encoder.parameters(), lr=params.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.n_episodes, eta_min=1e-6
    )
    encoder.train()

    train_history = {"episode": [], "loss": [], "accuracy": [], "lr": []}
    eval_history = {"episode": [], "loss": [], "accuracy": []}

    best_val_acc = 0.0
    best_val_loss = float('inf')
    no_improve_counter = 0

    for episode in range(params.n_episodes):
        # Training
        support_x, support_y, query_x, query_y = create_episode(
            train_set, n_way=params.n_way, k_shot=params.k_shot, q_query=params.q_query
        )
        loss, acc = quantum_inspired_prototypical_loss(encoder, support_x, support_y, query_x, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log training metrics
        train_history["episode"].append(episode)
        train_history["loss"].append(loss.item())
        train_history["accuracy"].append(acc.item())
        train_history["lr"].append(scheduler.get_last_lr()[0])

        # Print progress
        if episode % 100 == 0:
            print(
                f"Episode {episode}/{params.n_episodes} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {acc.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Time: {(time.time() - start_time) / 60:.2f} min"
            )

        # Evaluation
        if (episode + 1) % params.eval_every == 0 or episode == params.n_episodes - 1:
            encoder.eval()
            val_loss, val_acc = evaluate_model(encoder, val_set, params, n_episodes=100)
            encoder.train()

            # Log validation metrics
            eval_history["episode"].append(episode)
            eval_history["loss"].append(val_loss)
            eval_history["accuracy"].append(val_acc)

            print(
                f"Validation | "
                f"Loss: {val_loss:.4f} | "
                f"Accuracy: {val_acc:.4f}"
            )

            # Save best model
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                no_improve_counter = 0
                
                if params.best_model_path:
                    torch.save(encoder.state_dict(), params.best_model_path)
                    print(f"Saved best model with val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}")
            else:
                no_improve_counter += 1
                if no_improve_counter >= 10:  # Early stopping after 10 evaluations without improvement
                    print(f"Early stopping at episode {episode}")
                    break

    # Save training and evaluation logs
    if params.save_train_log:
        import pandas as pd
        pd.DataFrame(train_history).to_csv(params.save_train_log, index=False)
    
    if params.save_eval_log:
        import pandas as pd
        pd.DataFrame(eval_history).to_csv(params.save_eval_log, index=False)

    return train_history, eval_history

def evaluate_model(encoder, dataset, params, n_episodes=1000, log_path=None, cm_path=None):
    """Evaluate model on dataset."""
    encoder.eval()
    
    total_loss = 0
    total_acc = 0
    
    # Confusion matrix
    n_classes = params.n_way
    confusion_matrix = torch.zeros(n_classes, n_classes)
    
    with torch.no_grad():
        for episode in range(n_episodes):
            support_x, support_y, query_x, query_y = create_episode(
                dataset, n_way=params.n_way, k_shot=params.k_shot, q_query=params.q_query
            )
            
            # Forward pass
            support_embeddings = encoder(support_x)
            query_embeddings = encoder(query_x)
            
            # Compute prototypes
            prototypes, proto_labels = compute_prototypes(support_embeddings, support_y)
            
            # Calculate quantum fidelity
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            inner_products = torch.mm(query_norm, proto_norm.t())
            fidelities = inner_products.pow(2)
            log_p_y = F.log_softmax(fidelities * 10.0, dim=1)
            
            target_inds = torch.tensor([torch.where(proto_labels == y)[0].item() for y in query_y])
            loss = F.nll_loss(-log_p_y, target_inds.to(DEVICE))
            
            # Compute accuracy
            pred_inds = log_p_y.argmax(dim=1)
            acc = (pred_inds == target_inds.to(DEVICE)).float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
            
            # Update confusion matrix
            for t, p in zip(target_inds.to('cpu'), pred_inds.to('cpu')):
                confusion_matrix[t, p] += 1
    
    # Normalize confusion matrix
    for i in range(n_classes):
        confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()
    
    # Save confusion matrix
    if cm_path:
        import pandas as pd
        cm_df = pd.DataFrame(confusion_matrix.numpy())
        cm_df.to_csv(cm_path, index=False)
    
    # Compute average metrics
    avg_loss = total_loss / n_episodes
    avg_acc = total_acc / n_episodes
    
    # Save evaluation log
    if log_path:
        import pandas as pd
        pd.DataFrame({
            "loss": [avg_loss],
            "accuracy": [avg_acc]
        }).to_csv(log_path, index=False)
    
    return avg_loss, avg_acc

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    transform = MelSpectrogram(sample_rate=SAMPLE_RATE, **mel_kwargs)
    dataset = BreathingSegmentDataset("audio", "clean_label", transform=transform)
    
    # Use the quantum-inspired encoder
    encoder = QuantumInspiredEncoder(
        input_dim=64, 
        hidden_dim=128, 
        out_dim=64, 
        n_layers=3
    ).to(DEVICE)

    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Training set size: {len(train_set)} samples")
    print(f"Validation set size: {len(val_set)} samples")
    print(f"Test set size: {len(test_set)} samples")

    params = TrainingParams(
        n_episodes=30000,
        eval_every=1000,
        save_train_log="quantum_inspired_training_log.csv",
        save_eval_log="quantum_inspired_eval_log.csv",
        best_model_path="best_quantum_inspired_model.pt"
    )

    train_quantum_protonet_with_eval(train_set, val_set, encoder, params)

    # Load best model
    encoder.load_state_dict(torch.load(params.best_model_path))
    encoder.to(DEVICE)
    encoder.eval()

    # Evaluate
    evaluate_model(
        encoder,
        test_set,
        params,
        n_episodes=1000,
        log_path="quantum_inspired_test_eval_log.csv",
        cm_path="quantum_inspired_test_confusion_matrix.csv"
    )
