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
SEGMENT_LENGTH = 5 * SAMPLE_RATE

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
    n_way: int = 50
    k_shot: int = 3
    q_query: int = 5
    k_clusters=2
    gamma=1.0
    lam=0.5
    learning_rate: float = 1e-4
    eval_every: int = 10
    scheduler_step: int = 2000       
    scheduler_gamma: float = 0.5
    patience: int = 100
    save_train_log: str = "training_log.csv"
    save_eval_log: str = "eval_log.csv"
    best_model_path = "best_model.pt"


# ------------------ Utils ------------------


# ----------- ESC-50 Dataset Loader -----------
class ESC50Dataset(Dataset):
    def __init__(self, audio_dir, csv_path, folds=[1,2,3,4,5]):
        self.audio_dir = audio_dir
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['fold'].isin(folds)]
        self.df = self.df.reset_index(drop=True)
        self.sample_rate = SAMPLE_RATE  # VGGish expects 16kHz mono

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row['filename'])
        waveform, sr = torchaudio.load(wav_path)
        # Resample to 16kHz mono if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # VGGish expects numpy float32
        wav_np = waveform.squeeze().numpy().astype(np.float32)
        label = int(row['target'])
        return wav_np, label

    def __len__(self):
        return len(self.df)

# ------------------ Encoder ------------------
class VGGishEncoder(nn.Module):
    def __init__(self, out_dim=128, device='cpu'):
        super().__init__()
        # Initialize VGGish
        self.vggish = vggish(postprocess=False)
        self.vggish.eval()
        
        # Move VGGish to device first
        self.vggish.to(device)
        
        # Freeze VGGish parameters
        for param in self.vggish.parameters():
            param.requires_grad = False
            
        # Trainable projection
        self.proj = nn.Linear(128, out_dim)
        self.proj.to(device)
        self.device = device

    def forward(self, x_list):
        batch_embs = []
        for wav in x_list:
            examples = vggish_input.waveform_to_examples(wav, SAMPLE_RATE)
            examples = examples.float().to(self.device)  # Ensure examples are on correct device
            
            with torch.no_grad():
                emb = self.vggish(examples)
                # Ensure the output is on the correct device
                emb = emb.to(self.device).mean(dim=0)  # (128,)
            batch_embs.append(emb)
            
        embeddings = torch.stack(batch_embs)  # (batch, 128)
        return self.proj(embeddings)  # Project to out_dim


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
def compute_prototypes(embeddings, labels):
    """Compute class prototypes by averaging embeddings of each class"""
    device = embeddings.device
    classes = torch.unique(labels)
    prototypes = []
    
    for cls in classes:
        # Get all embeddings for this class
        class_mask = (labels == cls)
        class_embeddings = embeddings[class_mask]
        # Compute prototype as mean of class embeddings
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)
    
    prototypes = torch.stack(prototypes)  # Shape: [n_classes, embedding_dim]
    return prototypes, classes


def compute_prototypes_kmeans(embeddings, labels, k_clusters_per_class=2):
    device = embeddings.device
    classes = torch.unique(labels)
    all_prototypes = []
    proto_class_map = []
    for cls in classes:
        idx = (labels == cls)
        class_emb = embeddings[idx].detach().cpu().numpy()
        n_proto = min(k_clusters_per_class, len(class_emb))
        if n_proto == 1:
            proto = class_emb.mean(axis=0, keepdims=True) 
        else:
            kmeans = KMeans(n_clusters=n_proto, random_state=42, n_init=10)
            kmeans.fit(class_emb)
            proto = kmeans.cluster_centers_  
        all_prototypes.append(torch.from_numpy(proto).float())  # Convert back to tensor
        proto_class_map += [cls.item()] * proto.shape[0]
    
    all_prototypes = torch.cat(all_prototypes, dim=0).to(device)  # Move to correct device
    proto_class_map = torch.tensor(proto_class_map, device=device)
    return all_prototypes, proto_class_map

def prototypical_loss(embeddings, labels, prototypes, prototype_classes):
    """Standard prototypical network loss"""
    # Compute distances from each query to each prototype
    distances = torch.cdist(embeddings, prototypes)  # [n_queries, n_classes]
    
    # Convert distances to logits (negative distances)
    logits = -distances
    
    # Create target indices for each query
    targets = []
    for label in labels:
        # Find which prototype index corresponds to this label
        target_idx = (prototype_classes == label).nonzero(as_tuple=True)[0][0]
        targets.append(target_idx)
    
    targets = torch.tensor(targets, device=embeddings.device)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, targets)
    
    # Compute accuracy
    predictions = logits.argmax(dim=1)
    predicted_classes = prototype_classes[predictions]
    accuracy = (predicted_classes == labels).float().mean()
    
    return loss, accuracy

# def prototypical_loss(embeddings, labels, prototypes, proto_class_map):
#     distances = torch.cdist(embeddings, prototypes)
#     logits = -distances  # Convert distances to logits
    
#     # Create targets for cross-entropy
#     targets = []
#     for i, label in enumerate(labels):
#         # Find prototype indices for this class
#         class_protos = (proto_class_map == label).nonzero(as_tuple=True)[0]
#         # Use closest prototype of correct class
#         closest_proto = distances[i][class_protos].argmin()
#         targets.append(class_protos[closest_proto])
    
#     targets = torch.stack(targets)
#     loss = F.cross_entropy(logits, targets)
    
#     preds = logits.argmax(dim=1)
#     pred_classes = proto_class_map[preds]
#     acc = (pred_classes == labels).float().mean()
    
#     return loss, acc

def corel_loss(embeddings, labels, prototypes, proto_class_map, gamma=1.0, lam=0.5):
    batch_size = embeddings.size(0)
    
    # Compute all pairwise distances at once
    distances = torch.cdist(embeddings, prototypes)  # (batch_size, num_prototypes)
    
    losses = []
    preds = []
    
    for i in range(batch_size):
        label = labels[i]
        dists = distances[i]  # Distances from sample i to all prototypes
        
        # Positive distances (same class)
        pos_mask = (proto_class_map == label)
        pos_dists = dists[pos_mask]
        min_pos_dist = pos_dists.min()
        attractive = gamma * min_pos_dist
        
        # Negative distances (different classes)
        neg_mask = (proto_class_map != label)
        if neg_mask.sum() > 0:
            neg_dists = dists[neg_mask]
            # This encourages maximizing distance to negative prototypes
            repulsive_raw = gamma * torch.sum(torch.exp(-neg_dists))
            repulsive = repulsive_raw / neg_mask.sum()
        else:
            repulsive = 0.0
            
        loss = lam * attractive + (1 - lam) * repulsive
        losses.append(loss)
        
        # Prediction
        nearest_proto = dists.argmin()
        pred_class = proto_class_map[nearest_proto]
        preds.append(pred_class)
    
    preds = torch.stack(preds)
    acc = (preds == labels).float().mean()
    return torch.stack(losses).mean(), acc

# ------------------ Episode Generator ------------------
def create_episode(dataset, n_way=5, k_shot=5, q_query=5):
    df = dataset.df
    classes = np.random.choice(df['target'].unique(), n_way, replace=False)
    support_idx, query_idx = [], []
    for cls in classes:
        idx = df[df['target'] == cls].index.tolist()
        chosen = random.sample(idx, k_shot + q_query)
        support_idx += chosen[:k_shot]
        query_idx += chosen[k_shot:]
    support_x = [dataset[i][0] for i in support_idx]
    support_y = torch.tensor([dataset[i][1] for i in support_idx])
    query_x = [dataset[i][0] for i in query_idx]
    query_y = torch.tensor([dataset[i][1] for i in query_idx])
    return support_x, support_y, query_x, query_y

# ------------------ Training Loop ------------------
def train_protonet_with_eval(train_set, val_set, encoder, params: TrainingParams):
    start_time = time.time()

    optimizer = torch.optim.Adam(encoder.parameters(), lr=params.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params.scheduler_step, gamma=params.scheduler_gamma
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
        support_emb = encoder(support_x).to(DEVICE)
        query_emb = encoder(query_x).to(DEVICE)
        support_y = support_y.to(DEVICE)
        query_y = query_y.to(DEVICE)
        prototypes, proto_class_map = compute_prototypes(support_emb, support_y)
        loss, acc = prototypical_loss(query_emb, query_y, prototypes, proto_class_map)

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
                support_emb = encoder(support_x).to(DEVICE)
                query_emb = encoder(query_x).to(DEVICE)
                support_y = support_y.to(DEVICE)
                query_y = query_y.to(DEVICE)
                prototypes, proto_class_map = compute_prototypes(support_emb, support_y)
                loss, acc = prototypical_loss(query_emb, query_y, prototypes, proto_class_map)
                dists = torch.cdist(query_emb, prototypes)
                nearest_proto = dists.argmin(dim=1)
                pred = proto_class_map[nearest_proto]
                acc = (pred == query_y).float().mean()

                eval_history["episode"].append(episode)
                eval_history["loss"].append(loss.item())
                eval_history["accuracy"].append(acc.item())
                print(f"[Eval]  Episode {episode}: Val Loss={loss.item():.4f}, Val Acc={acc.item():.4f}")

                if acc.item() > best_val_acc:
                    best_val_acc = acc.item()

                if loss.item() < best_val_loss:
                    best_val_loss = loss.item()
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
            support_emb = encoder(support_x).to(DEVICE)
            query_emb = encoder(query_x).to(DEVICE)
            support_y = support_y.to(DEVICE)
            query_y = query_y.to(DEVICE)
            prototypes, proto_class_map = compute_prototypes(support_emb, support_y)
            dists = torch.cdist(query_emb, prototypes)
            nearest_proto = dists.argmin(dim=1)
            pred = proto_class_map[nearest_proto]
            acc = (pred == query_y).float().mean().item()
            accs.append(acc)

            all_preds.extend(pred.cpu().numpy())
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

def evaluate_with_full_prototypes(encoder, train_dataset, test_dataset, params, 
                                log_path="full_prototype_eval_log.csv", 
                                cm_path="full_prototype_confusion_matrix.csv"):
    """
    Evaluate using full prototypes computed from entire training dataset
    """
    encoder.eval()
    
    print("Computing full prototypes from training dataset...")
    
    # Step 1: Compute embeddings for entire training dataset
    all_train_embeddings = []
    all_train_labels = []
    
    with torch.no_grad():
        # Process training data in batches to avoid memory issues
        batch_size = 32
        for i in tqdm(range(0, len(train_dataset), batch_size), desc="Processing training data"):
            batch_x = []
            batch_y = []
            
            end_idx = min(i + batch_size, len(train_dataset))
            for j in range(i, end_idx):
                x, y = train_dataset[j]
                batch_x.append(x)
                batch_y.append(y)
            
            # Get embeddings for this batch
            batch_embeddings = encoder(batch_x).to(DEVICE)
            batch_labels = torch.tensor(batch_y).to(DEVICE)
            
            all_train_embeddings.append(batch_embeddings)
            all_train_labels.append(batch_labels)
    
    # Concatenate all embeddings and labels
    all_train_embeddings = torch.cat(all_train_embeddings, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)
    
    full_prototypes, prototype_classes = compute_prototypes(all_train_embeddings, all_train_labels)
    print(f"Computed {len(prototype_classes)} prototypes from {len(all_train_embeddings)} training samples")
    
    # Step 3: Evaluate on test dataset
    print("Evaluating on test dataset...")
    all_preds = []
    all_labels = []
    all_distances = []
    
    with torch.no_grad():
        # Process test data in batches
        batch_size = 32
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing test data"):
            batch_x = []
            batch_y = []
            
            end_idx = min(i + batch_size, len(test_dataset))
            for j in range(i, end_idx):
                x, y = test_dataset[j]
                batch_x.append(x)
                batch_y.append(y)
            
            # Get embeddings for this batch
            batch_embeddings = encoder(batch_x).to(DEVICE)
            batch_labels = torch.tensor(batch_y).to(DEVICE)
            
            # Compute distances to all prototypes
            distances = torch.cdist(batch_embeddings, full_prototypes)  # [batch_size, n_classes]
            
            # Get predictions (closest prototype)
            nearest_proto_indices = distances.argmin(dim=1)
            batch_preds = prototype_classes[nearest_proto_indices]
            
            all_preds.extend(batch_preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_distances.extend(distances.min(dim=1)[0].cpu().numpy())
    
    # Step 4: Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    print(f"\nFull Prototype Evaluation Results:")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, output_dict=True)
    verbose_report = classification_report(all_labels, all_preds)
    print(verbose_report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Step 5: Save results
    # Save summary log
    log_data = {
        "overall_accuracy": [accuracy],
        "n_test_samples": [len(all_labels)],
        "n_train_samples": [len(all_train_embeddings)],
        "n_prototypes": [len(prototype_classes)]
    }
    
    # Add per-class metrics
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                log_data[f"{label}_{metric}"] = [value]
    
    pd.DataFrame(log_data).to_csv(log_path, index=False)
    print(f"Evaluation log saved to {log_path}")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(cm_path, index=False)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save detailed results
    results_path = "full_prototype_results.txt"
    with open(results_path, "w") as f:
        f.write("=== Full Prototype Evaluation Results ===\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Number of test samples: {len(all_labels)}\n")
        f.write(f"Number of training samples used: {len(all_train_embeddings)}\n")
        f.write(f"Number of prototypes: {len(prototype_classes)}\n\n")
        f.write("=== Classification Report ===\n")
        f.write(str(verbose_report))
        f.write("\n\n=== Confusion Matrix ===\n")
        f.write(str(cm))
    print(f"Detailed results saved to {results_path}")
    
    return accuracy, report, cm


# ------------------ Run ------------------
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    AUDIO_DIR = "./ESC-50-master/audio"
    CSV_PATH = "./ESC-50-master/meta/esc50.csv"
    train_set = ESC50Dataset(AUDIO_DIR, CSV_PATH, folds=[1,2,3,4])
    val_set = ESC50Dataset(AUDIO_DIR, CSV_PATH, folds=[5])
    test_set = ESC50Dataset(AUDIO_DIR, CSV_PATH, folds=[5])

    # encoder = CNNEncoder(mel_kwargs=mel_kwargs).to(DEVICE)
    encoder = VGGishEncoder(out_dim=64, device=DEVICE)

    total_len = len(train_set) + len(val_set)
    print(f"Dataset size: {total_len} samples")
    print(f"Training set size: {len(train_set)} samples")
    print(f"Validation set size: {len(val_set)} samples")
    print(f"Test set size: {len(test_set)} samples")

    params = TrainingParams(
        n_episodes=10000,
        eval_every=20,
        save_train_log="training_log.csv",
        save_eval_log="eval_log.csv"
    )

    # Replace the training call
    train_protonet_with_eval(train_set, val_set, encoder, params)
    
    # Load best model
    encoder.load_state_dict(torch.load(params.best_model_path, weights_only=True))
    encoder.to(DEVICE)
    encoder.eval()

    # Evaluate
    evaluate_model(
        encoder,
        test_set,
        params,
        n_episodes=1000,
        log_path="test_eval_log_kmeans.csv",
        cm_path="test_confusion_matrix_kmeans.csv"
    )

    evaluate_with_full_prototypes(
        encoder,
        train_set,  # Use entire training set to compute prototypes
        test_set,   # Evaluate on test set
        params,
        log_path="full_prototype_eval_log.csv",
        cm_path="full_prototype_confusion_matrix.csv"
    )
