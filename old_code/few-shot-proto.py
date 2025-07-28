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
from sklearn.cluster import KMeans

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

@dataclass
class TrainingParams:
    n_episodes: int = 10000
    n_way: int = 5
    k_shot: int = 10
    q_query: int = 15
    learning_rate: float = 1e-4
    eval_every: int = 10
    scheduler_step: int = 2000       
    scheduler_gamma: float = 0.5
    patience: int = 100
    save_train_log: str = "training_log.csv"
    save_eval_log: str = "eval_log.csv"
    best_model_path = "best_model.pt"


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


# ------------------ Encoder ------------------
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=64, sample_rate=SAMPLE_RATE, segment_length=SEGMENT_LENGTH, mel_kwargs=None):
        super().__init__()
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            **mel_kwargs,
        )
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        # Dynamically compute flattened size using the same spectrogram transform
        with torch.no_grad():
            dummy_waveform = torch.zeros(1, segment_length)
            mel_spec = self.mel_transform(dummy_waveform)  # [1, n_mels, time]
            mel_spec = mel_spec.unsqueeze(0)  # Add batch and channel dims: [1, 1, n_mels, time]
            conv_out = self.conv_layers(mel_spec)
            self.flatten_dim = conv_out.view(1, -1).shape[1]

        self.fc = nn.Linear(self.flatten_dim, out_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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



# ------------------ Prototypical Network ------------------
def compute_prototypes(support, support_labels):
    classes = torch.unique(support_labels)
    prototypes = []
    for cls in classes:
        cls_embeddings = support[support_labels == cls]
        prototypes.append(cls_embeddings.mean(dim=0))
    return torch.stack(prototypes), classes

def compute_prototypes_kmeans(support, support_labels, prototypes_per_class=1):
    """
    Compute prototypes using K-means clustering instead of simple averaging
    """
    classes = torch.unique(support_labels)
    prototypes = []
    prototype_labels = []
    
    for cls in classes:
        cls_embeddings = support[support_labels == cls]
        
        # Convert to numpy for sklearn
        cls_embeddings_np = cls_embeddings.detach().cpu().numpy()
        
        if len(cls_embeddings_np) < prototypes_per_class:
            # If we have fewer samples than desired prototypes, just use the mean
            prototypes.append(cls_embeddings.mean(dim=0))
            prototype_labels.append(cls)
        else:
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=prototypes_per_class, random_state=42, n_init=10)
            kmeans.fit(cls_embeddings_np)
            
            # Convert centroids back to torch tensors
            centroids = torch.tensor(kmeans.cluster_centers_, 
                                   dtype=support.dtype, device=support.device)
            
            for centroid in centroids:
                prototypes.append(centroid)
                prototype_labels.append(cls)
    
    return torch.stack(prototypes), torch.tensor(prototype_labels, device=support.device)


def prototypical_loss(encoder, support, support_labels, query, query_labels):
    support_embeddings = encoder(support)
    query_embeddings = encoder(query)

    prototypes, proto_labels = compute_prototypes(support_embeddings, support_labels)

    dists = torch.cdist(query_embeddings, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)
    target_inds = torch.tensor([torch.where(proto_labels == y)[0].item() for y in query_labels])
    loss = F.nll_loss(log_p_y, target_inds.to(DEVICE))
    acc = (log_p_y.argmax(dim=1) == target_inds.to(DEVICE)).float().mean()
    return loss, acc

def prototypical_loss_kmeans(encoder, support, support_labels, query, query_labels, prototypes_per_class=1):
    support_embeddings = encoder(support)
    query_embeddings = encoder(query)

    # Use K-means for prototype computation
    prototypes, proto_labels = compute_prototypes_kmeans(
        support_embeddings, support_labels, prototypes_per_class
    )

    # Calculate distances from queries to all prototypes
    dists = torch.cdist(query_embeddings, prototypes)
    
    # For each query, find the nearest prototype and assign its class
    nearest_prototype_indices = torch.argmin(dists, dim=1)
    predicted_labels = proto_labels[nearest_prototype_indices]
    
    # Calculate accuracy
    acc = (predicted_labels == query_labels).float().mean()
    
    # Simplified loss: for each query, find minimum distance to correct class prototypes
    losses = []
    unique_classes = torch.unique(proto_labels)
    
    for i, true_label in enumerate(query_labels):
        # Find all prototypes belonging to the true class
        correct_class_mask = (proto_labels == true_label)
        
        if correct_class_mask.sum() > 0:
            # Get distances to all prototypes
            query_dists = dists[i]
            
            # Apply log softmax to negative distances
            log_probs = F.log_softmax(-query_dists, dim=0)
            
            # Sum log probabilities for correct class prototypes
            correct_class_log_prob = torch.logsumexp(log_probs[correct_class_mask], dim=0)
            losses.append(-correct_class_log_prob)
        else:
            # Fallback: use cross entropy with nearest prototype
            log_probs = F.log_softmax(-query_dists, dim=0)
            losses.append(-log_probs[nearest_prototype_indices[i]])
    
    loss = torch.stack(losses).mean()
    return loss, acc


# ------------------ Episode Generator ------------------
def create_episode(dataset, n_way=5, k_shot=5, q_query=5):
    # Get available classes in the dataset
    available_classes = set([x[1] for x in dataset])
    # Ensure we don't try to sample more classes than available
    n_way = min(n_way, len(available_classes))
    classes = random.sample(list(available_classes), n_way)
    
    support, query = [], []
    for cls in classes:
        items = [x for x in dataset if x[1] == cls]
        if len(items) < k_shot + q_query:
            # If not enough samples, sample with replacement
            chosen = random.choices(items, k=k_shot + q_query)
        else:
            chosen = random.sample(items, k_shot + q_query)
        support.extend(chosen[:k_shot])
        query.extend(chosen[k_shot:])
    
    support_x = torch.stack([x[0] for x in support]).unsqueeze(1).to(DEVICE)
    support_y = torch.tensor([x[1] for x in support]).to(DEVICE)
    query_x = torch.stack([x[0] for x in query]).unsqueeze(1).to(DEVICE)
    query_y = torch.tensor([x[1] for x in query]).to(DEVICE)
    return support_x, support_y, query_x, query_y

# ------------------ Training Loop ------------------
def train_protonet(dataset, encoder, params: TrainingParams):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=params.learning_rate)
    encoder.train()

    history = {"episode": [], "loss": [], "accuracy": []}

    for episode in range(params.n_episodes):
        support_x, support_y, query_x, query_y = create_episode(
            dataset, n_way=params.n_way, k_shot=params.k_shot, q_query=params.q_query
        )
        loss, acc = prototypical_loss(encoder, support_x, support_y, query_x, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["episode"].append(episode)
        history["loss"].append(loss.item())
        history["accuracy"].append(acc.item())

        if episode % 50 == 0:
            print(f"[Train] Episode {episode}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")

    pd.DataFrame(history).to_csv(params.save_train_log, index=False)
    print(f"Training log saved to {params.save_train_log}")

def train_protonet_with_eval(train_set, val_set, encoder, params: TrainingParams):
    start_time = time.time()

    optimizer = torch.optim.Adam(encoder.parameters(), lr=params.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.n_episodes, eta_min=1e-6
    )
    encoder.train()

    train_history = {"episode": [], "loss": [], "accuracy": [], "lr": []}
    eval_history = {"episode": [], "loss": [], "accuracy": []}

    best_val_acc = 0.0
    best_val_loss = float('inf')  # Start with infinity
    no_improve_counter = 0

    for episode in range(params.n_episodes):
        # Training
        support_x, support_y, query_x, query_y = create_episode(
            train_set, n_way=params.n_way, k_shot=params.k_shot, q_query=params.q_query
        )
        loss, acc = prototypical_loss(encoder, support_x, support_y, query_x, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        train_history["episode"].append(episode)
        train_history["loss"].append(loss.item())
        train_history["accuracy"].append(acc.item())
        train_history["lr"].append(current_lr)

        print(f"[Train] Episode {episode}: Loss={loss.item():.4f}, Acc={acc.item():.4f}, LR={current_lr:.6f}")

        # Evaluation
        if (episode + 1) % params.eval_every == 0:
            encoder.eval()
            with torch.no_grad():
                support_x, support_y, query_x, query_y = create_episode(
                    val_set, n_way=params.n_way, k_shot=params.k_shot, q_query=params.q_query
                )
                val_loss, val_acc = prototypical_loss(encoder, support_x, support_y, query_x, query_y)
                eval_history["episode"].append(episode)
                eval_history["loss"].append(val_loss.item())
                eval_history["accuracy"].append(val_acc.item())
                print(f"[Eval]  Episode {episode}: Val Loss={val_loss.item():.4f}, Val Acc={val_acc.item():.4f}")

                if val_acc.item() > best_val_acc:
                    best_val_acc = val_acc.item()

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    no_improve_counter = 0
                    torch.save(encoder.state_dict(), params.best_model_path)
                    print(f"Saved new best model at episode {episode} with val loss {best_val_loss:.4f}")
                else:
                    no_improve_counter += 1
                    print(f"No improvement. Counter: {no_improve_counter}/{params.patience}")

                # Early stopping check
                if no_improve_counter >= params.patience:
                    print(f"Early stopping at episode {episode} (val loss stopped improving)")
                    break

            encoder.train()

    # Save logs
    pd.DataFrame(train_history).to_csv(params.save_train_log, index=False)
    pd.DataFrame(eval_history).to_csv(params.save_eval_log, index=False)
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Best model saved to {params.best_model_path} with val acc {best_val_acc:.4f}, val loss {best_val_loss:.4f}")
    with open('train_log.txt', 'w') as f:
        f.write(f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")


def train_protonet_kmeans(train_dataset, test_dataset, encoder, params: TrainingParams, prototypes_per_class=2):
    start_time = time.time()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=params.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.n_episodes, eta_min=1e-6
    )

    train_history = {"episode": [], "loss": [], "accuracy": [], "lr": []}
    eval_history = {"episode": [], "loss": [], "accuracy": []}

    best_val_acc = 0.0
    best_val_loss = float('inf')  # Start with infinity
    no_improve_counter = 0
    
    for episode in range(params.n_episodes):
        # Training phase
        encoder.train()
        support_x, support_y, query_x, query_y = create_episode(
            train_dataset, n_way=params.n_way, k_shot=params.k_shot, q_query=params.q_query
        )
        loss, train_acc = prototypical_loss_kmeans(
            encoder, support_x, support_y, query_x, query_y, prototypes_per_class
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"[Train] Episode {episode}: Loss={loss.item():.4f}, Acc={train_acc.item():.4f}, LR={current_lr:.6f}")

        if (episode + 1) % params.eval_every == 0:
            # Evaluation phase
            encoder.eval()
            with torch.no_grad():
                test_support_x, test_support_y, test_query_x, test_query_y = create_episode(
                    test_dataset, n_way=params.n_way, k_shot=params.k_shot, q_query=params.q_query
                )
                val_loss, val_acc = prototypical_loss_kmeans(
                    encoder, test_support_x, test_support_y, test_query_x, test_query_y, prototypes_per_class
                )
            
            eval_history["episode"].append(episode)
            eval_history["loss"].append(val_loss.item())
            eval_history["accuracy"].append(val_acc.item())
            print(f"[Eval]  Episode {episode}: Val Loss={val_loss.item():.4f}, Val Acc={val_acc.item():.4f}")

            if val_acc.item() > best_val_acc:
                best_val_acc = val_acc.item()

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                no_improve_counter = 0
                torch.save(encoder.state_dict(), params.best_model_path)
                print(f"Saved new best model at episode {episode} with val loss {best_val_loss:.4f}")
            else:
                no_improve_counter += 1
                print(f"No improvement. Counter: {no_improve_counter}/{params.patience}")

            # Early stopping check
            if no_improve_counter >= params.patience:
                print(f"Early stopping at episode {episode} (val loss stopped improving)")
                break

        # Record history
        train_history["episode"].append(episode)
        train_history["loss"].append(loss.item())
        train_history["accuracy"].append(train_acc.item())
        train_history["lr"].append(current_lr)

    # Save logs
    pd.DataFrame(train_history).to_csv(params.save_train_log, index=False)
    pd.DataFrame(eval_history).to_csv(params.save_eval_log, index=False)
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Best model saved to {params.best_model_path} with val acc {best_val_acc:.4f}, val loss {best_val_loss:.4f}")
    with open('train_log.txt', 'w') as f:
        f.write(f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")


# ------------------ Evaluate ------------------
def evaluate_model(encoder, dataset, params, n_episodes=100, 
                   log_path="test_eval_log.csv", cm_path="test_confusion_matrix.csv"):
    encoder.eval()
    accs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for _ in tqdm(range(n_episodes), desc="Evaluating"):
            support_x, support_y, query_x, query_y = create_episode(
                dataset,
                n_way=params.n_way,
                k_shot=params.k_shot,
                q_query=params.q_query
            )

            support_emb = encoder(support_x)
            query_emb = encoder(query_x)

            prototypes, proto_labels = compute_prototypes(support_emb, support_y)
            dists = torch.cdist(query_emb, prototypes)
            log_p_y = F.log_softmax(-dists, dim=1)
            preds = log_p_y.argmax(dim=1)
            pred_labels = proto_labels[preds]

            acc = (pred_labels == query_y).float().mean().item()
            accs.append(acc)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(query_y.cpu().numpy())

    # === Metrics ===
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"\nFinal Evaluation on Test Set ({n_episodes} episodes):")
    print(f"Mean Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, output_dict=True)
    verbose_report = classification_report(all_labels, all_preds)
    print(verbose_report)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # === Save Summary Log ===
    log_data = {
        "mean_accuracy": [mean_acc],
        "std_accuracy": [std_acc],
        "n_episodes": [n_episodes]
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
        f.write("=== Classification Report ===\n")
        f.write(f"Mean Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%\n\n")
        f.write(str(verbose_report))
    print(f"Classification report saved to {results_path}")

def evaluate_model_kmeans(encoder, dataset, params, prototypes_per_class=2, n_episodes=100, 
                         log_path="test_eval_log.csv", cm_path="test_confusion_matrix.csv"):
    encoder.eval()
    accs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for _ in tqdm(range(n_episodes), desc="Evaluating with K-means"):
            support_x, support_y, query_x, query_y = create_episode(
                dataset,
                n_way=params.n_way,
                k_shot=params.k_shot,
                q_query=params.q_query
            )

            support_emb = encoder(support_x)
            query_emb = encoder(query_x)

            # Use K-means prototypes
            prototypes, proto_labels = compute_prototypes_kmeans(
                support_emb, support_y, prototypes_per_class
            )
            
            # Classify queries based on nearest prototype
            dists = torch.cdist(query_emb, prototypes)
            nearest_prototype_indices = torch.argmin(dists, dim=1)
            pred_labels = proto_labels[nearest_prototype_indices]

            acc = (pred_labels == query_y).float().mean().item()
            accs.append(acc)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(query_y.cpu().numpy())

    # === Metrics ===
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"\nFinal Evaluation on Test Set ({n_episodes} episodes):")
    print(f"Mean Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, output_dict=True)
    verbose_report = classification_report(all_labels, all_preds)
    print(verbose_report)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # === Save Summary Log ===
    log_data = {
        "mean_accuracy": [mean_acc],
        "std_accuracy": [std_acc],
        "n_episodes": [n_episodes]
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
        f.write("=== Classification Report ===\n")
        f.write(f"Mean Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%\n\n")
        f.write(str(verbose_report))
    print(f"Classification report saved to {results_path}")


# ------------------ Run ------------------
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    transform = MelSpectrogram(sample_rate=SAMPLE_RATE, **mel_kwargs)
    dataset = BreathingSegmentDataset("audio", "strong_labels", transform=transform)
    # encoder = CNNEncoder(mel_kwargs=mel_kwargs).to(DEVICE)
    encoder = VGGishEncoder(out_dim=64).to(DEVICE)


    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

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

    params = TrainingParams(
        n_episodes=10000,
        eval_every=20,
        save_train_log="training_log.csv",
        save_eval_log="eval_log.csv"
    )

    # Use K-means training with multiple prototypes per class
    prototypes_per_class = 5  # You can experiment with this value
    
    # Replace the training call
    train_protonet_kmeans(train_set, val_set, encoder, params, prototypes_per_class)
    
    # Load best model
    encoder.load_state_dict(torch.load(params.best_model_path, weights_only=True))
    encoder.to(DEVICE)
    encoder.eval()

    # Evaluate
    evaluate_model_kmeans(
        encoder,
        test_set,
        params,
        prototypes_per_class=prototypes_per_class,
        n_episodes=1000,
        log_path="test_eval_log_kmeans.csv",
        cm_path="test_confusion_matrix_kmeans.csv"
    )
