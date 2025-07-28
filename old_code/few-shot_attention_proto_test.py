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

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

from pathlib import Path

# --- 1. Global Configuration & Hyperparameters ---

# Training Parameters
class TrainingParams:
    n_way = 5                 # N: Number of classes in an episode
    k_shot = 20               # K: Number of support samples per class
    q_query = 5               # Q: Number of query samples per class
    n_episodes = 10000         # Total training episodes
    eval_every = 50           # Evaluate on validation set every N episodes
    patience = 100             # Early stopping patience
    learning_rate = 0.00005
    n_clusters_per_class = 5  # M: Number of k-means clusters (sub-prototypes)
    embedding_dim = 64
    best_model_path = "./exp/20250615/1/best_hybrid_model.pth"
    train_log_path = "training_log.csv"
    eval_log_path = "validation_log.csv"
    final_report_path = "final_evaluation_report.txt"
    final_cm_path = "final_confusion_matrix.csv"

mel_kwargs = {
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 64,
}

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 10 * SAMPLE_RATE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
BREATHING_LABELS = {
    "breathing": 0,
    "light breathing": 1,
    "heavy breathing": 2,
    "heavy breathing with noises": 3,
    "no breathing": 4
}
# Create a reverse mapping for classification report
LABEL_NAMES = [k for k, v in sorted(BREATHING_LABELS.items(), key=lambda item: item[1])]

# --- 2. Data Loading Utilities & Dataset Class ---

def load_labels(label_path):
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            try:
                onset, offset, label = line.strip().split(maxsplit=2)
                labels.append((float(onset), float(offset), label.lower().strip()))
            except ValueError:
                print(f"Warning: Skipping malformed line in {label_path}: {line.strip()}")
    return labels

def is_breathing_segment(start_sec, end_sec, label_entries):
    for onset, offset, label in label_entries:
        if max(onset, start_sec) < min(offset, end_sec):
            best_match_length = 0
            best_class_idx = None
            for breathing_type, class_idx in BREATHING_LABELS.items():
                if breathing_type in label and len(breathing_type) > best_match_length:
                    best_match_length = len(breathing_type)
                    best_class_idx = class_idx
            if best_class_idx is not None:
                return best_class_idx
    return BREATHING_LABELS["no breathing"]

class BreathingSegmentDataset(Dataset):
    def __init__(self, audio_dir, label_dir, segment_length=SEGMENT_LENGTH, 
                 augment_minority_classes=True, target_min_samples=50):
        self.data = []
        self.targets = []
        self.n_classes = len(BREATHING_LABELS)
        self.augment_minority_classes = augment_minority_classes
        self.target_min_samples = target_min_samples
        
        print("Loading dataset...")
        self._load_original_data(audio_dir, label_dir, segment_length)
        
        if self.augment_minority_classes:
            print("Augmenting minority classes...")
            self._augment_data()
            
        self._print_class_distribution()

    def _load_original_data(self, audio_dir, label_dir, segment_length):
        """Load the original dataset without augmentation."""
        audio_files = glob.glob(f"{audio_dir}/**/*.wav", recursive=True)
        for audio_path in tqdm(audio_files, desc="Processing audio files"):
            file_id = Path(audio_path).stem
            label_path = os.path.join(label_dir, f"{file_id}.txt")
            if not os.path.exists(label_path):
                continue
            
            labels = load_labels(label_path)
            try:
                waveform, sr = torchaudio.load(audio_path)
                if sr != SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}, skipping...")
                continue

            num_segments = waveform.shape[1] // segment_length
            for i in range(num_segments):
                start = i * segment_length
                segment = waveform[:, start:start+segment_length]
                label = is_breathing_segment(start/SAMPLE_RATE, (start+segment_length)/SAMPLE_RATE, labels)
                
                self.data.append((segment, label))
                self.targets.append(label)

    def _augment_data(self):
        """Fixed augmentation with proper audio transforms."""
        import torchaudio.transforms as T
        
        # Count samples per class
        class_counts = {}
        for target in self.targets:
            class_counts[target] = class_counts.get(target, 0) + 1
        
        print("Original class distribution:")
        for class_name, class_idx in BREATHING_LABELS.items():
            count = class_counts.get(class_idx, 0)
            print(f"  {class_name}: {count} samples")
        
        # Define FIXED augmentation transforms that work with raw waveforms
        def add_noise(waveform, noise_factor=0.01):
            """Add Gaussian noise to waveform."""
            noise = torch.randn_like(waveform) * noise_factor
            return waveform + noise
        
        def change_volume(waveform, factor):
            """Change volume by a factor."""
            return torch.clamp(waveform * factor, -1.0, 1.0)
        
        def pitch_shift_simple(waveform, shift_steps=2):
            """Simple pitch shift using resampling."""
            try:
                pitch_shifter = T.PitchShift(sample_rate=SAMPLE_RATE, n_steps=shift_steps)
                return pitch_shifter(waveform)
            except:
                return waveform  # Return original if fails
        
        def time_stretch_fixed(waveform, rate=0.9):
            """Time stretch with proper STFT handling."""
            try:
                # Convert to STFT domain
                stft_transform = T.Spectrogram(
                    n_fft=1024, 
                    hop_length=256, 
                    power=None  # Keep complex values
                )
                time_stretch = T.TimeStretch(
                    hop_length=256, 
                    n_freq=513,  # (n_fft // 2) + 1
                    fixed_rate=rate
                )
                istft_transform = T.InverseSpectrogram(
                    n_fft=1024,
                    hop_length=256
                )
                
                # Apply time stretch in STFT domain
                stft = stft_transform(waveform)
                stretched_stft = time_stretch(stft)
                stretched_waveform = istft_transform(stretched_stft)
                
                # Ensure same length as original
                if stretched_waveform.shape[-1] > waveform.shape[-1]:
                    stretched_waveform = stretched_waveform[..., :waveform.shape[-1]]
                elif stretched_waveform.shape[-1] < waveform.shape[-1]:
                    padding = waveform.shape[-1] - stretched_waveform.shape[-1]
                    stretched_waveform = F.pad(stretched_waveform, (0, padding))
                
                return stretched_waveform
            except Exception as e:
                print(f"Time stretch failed: {e}")
                return waveform
        
        # List of working transforms
        transforms = [
            lambda x: add_noise(x, 0.01),
            lambda x: change_volume(x, 0.7),
            lambda x: change_volume(x, 1.3),
            lambda x: pitch_shift_simple(x, 2),
            lambda x: pitch_shift_simple(x, -2),
            lambda x: time_stretch_fixed(x, 0.9),
            lambda x: time_stretch_fixed(x, 1.1),
        ]
        
        # Augment each minority class
        for class_label, current_count in class_counts.items():
            if current_count < self.target_min_samples:
                print(f"Augmenting class {class_label} from {current_count} to {self.target_min_samples} samples...")
                
                # Find all samples of this class
                class_samples = [(i, (wav, lbl)) for i, (wav, lbl) in enumerate(self.data) 
                               if self.targets[i] == class_label]
                
                samples_needed = self.target_min_samples - current_count
                
                for _ in tqdm(range(samples_needed), desc=f"Augmenting class {class_label}"):
                    # Randomly select a sample and transform
                    idx, (original_wav, original_label) = random.choice(class_samples)
                    transform = random.choice(transforms)
                    
                    try:
                        # CRITICAL FIX: Create a deep copy to avoid autograd issues
                        original_wav_copy = original_wav.clone().detach()
                        augmented_wav = transform(original_wav_copy)
                        
                        # Ensure proper tensor properties
                        augmented_wav = augmented_wav.detach()
                        
                        self.data.append((augmented_wav, original_label))
                        self.targets.append(class_label)
                        
                    except Exception as e:
                        print(f"Augmentation failed: {e}. Using original sample.")
                        # Create a copy even for the fallback
                        original_copy = original_wav.clone().detach()
                        self.data.append((original_copy, original_label))
                        self.targets.append(class_label)

    def _print_class_distribution(self):
        """Print the final class distribution after augmentation."""
        class_counts = {}
        for target in self.targets:
            class_counts[target] = class_counts.get(target, 0) + 1
        
        print("\nFinal class distribution:")
        for class_name, class_idx in BREATHING_LABELS.items():
            count = class_counts.get(class_idx, 0)
            print(f"  {class_name}: {count} samples")
        print(f"Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, label = self.data[idx]
        # CRITICAL FIX: Always return a fresh copy to avoid autograd graph reuse
        return waveform.clone().detach(), label


# --- 3. Model Components ---
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, height, width))

    def forward(self, x):
        return x + self.pos_embedding

class AttentionEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=mel_kwargs["n_mels"], 
            n_fft=mel_kwargs["n_fft"], hop_length=mel_kwargs["hop_length"]
        )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        with torch.no_grad():
            dummy_out = self.conv_layers(self.mel_transform(torch.zeros(1, SEGMENT_LENGTH)).unsqueeze(1))
            _, c, h, w = dummy_out.shape
            d_model, h, w = c, h, w

        self.pos_encoder = PositionalEncoding2D(d_model, h, w)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.attention = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.fc = nn.Linear(d_model * h * w, out_dim)

    def forward(self, x):
        # The encoder now expects a raw waveform with a channel dimension
        if x.dim() == 2: # B, T -> B, 1, T
            x = x.unsqueeze(1)
        mel_spec = self.mel_transform(x.squeeze(1)).unsqueeze(1)
        features = self.conv_layers(mel_spec)
        features = self.pos_encoder(features)
        b, c, h, w = features.shape
        features_seq = features.flatten(2).permute(0, 2, 1)
        attention_out = self.attention(features_seq)
        embedding = self.fc(attention_out.reshape(b, -1))
        return embedding

class QueryAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, query_emb, support_or_proto_embs):
        k = support_or_proto_embs.shape[0]
        query_repeated = query_emb.unsqueeze(0).repeat(k, 1)
        combined = torch.cat([query_repeated, support_or_proto_embs], dim=1)
        scores = self.layer(combined).squeeze(1)
        weights = F.softmax(scores, dim=0)
        return weights


# --- 4. Helper Functions for Training ---

def get_cluster_prototypes(support_embs, n_clusters):
    """Enhanced clustering with adaptive cluster selection."""
    if len(support_embs) == 0: 
        return torch.tensor([]).to(support_embs.device)
    
    support_embs_np = support_embs.detach().cpu().numpy()
    
    # Adaptive cluster selection based on available unique samples
    unique_embeddings = np.unique(support_embs_np, axis=0)
    effective_n_clusters = min(n_clusters, len(unique_embeddings))
    
    if effective_n_clusters == 1:
        return torch.from_numpy(unique_embeddings).float().to(support_embs.device)
    
    try:
        kmeans = KMeans(
            n_clusters=effective_n_clusters, 
            random_state=0, 
            n_init='auto'
        ).fit(support_embs_np)
        centroids = torch.from_numpy(kmeans.cluster_centers_).float().to(support_embs.device)
        return centroids
    except Exception as e:
        print(f"K-means failed: {e}. Using mean as single prototype.")
        mean_prototype = torch.mean(support_embs, dim=0, keepdim=True)
        return mean_prototype

def create_episode(dataset, n_way, k_shot, q_query):
    """Creates episodes with improved error handling."""
    if isinstance(dataset, torch.utils.data.Subset):
        subset_indices = dataset.indices
        original_dataset = dataset.dataset
        targets = np.array(original_dataset.targets)[subset_indices]
    else:
        subset_indices = list(range(len(dataset)))
        original_dataset = dataset
        targets = np.array(original_dataset.targets)
    
    unique_classes_in_split = np.unique(targets)
    if len(unique_classes_in_split) < n_way:
        raise ValueError(f"Not enough unique classes in this dataset split to create a {n_way}-way episode. Found only {len(unique_classes_in_split)}.")
    
    episode_classes = np.random.choice(unique_classes_in_split, n_way, replace=False)
    
    support_x_list, support_y_list = [], []
    query_x_list, query_y_list = [], []

    for cls in episode_classes:
        positions_in_split = np.where(targets == cls)[0]
        absolute_class_indices = np.array(subset_indices)[positions_in_split]
        
        num_samples_needed = k_shot + q_query
        
        if len(absolute_class_indices) < num_samples_needed:
            replace = True
        else:
            replace = False
        
        selected_absolute_indices = np.random.choice(absolute_class_indices, num_samples_needed, replace=replace)
        
        for i, idx in enumerate(selected_absolute_indices):
            waveform, label = original_dataset[idx]
            
            if i < k_shot:
                support_x_list.append(waveform)
                support_y_list.append(label)
            else:
                query_x_list.append(waveform)
                query_y_list.append(label)
    
    return (torch.stack(support_x_list), torch.tensor(support_y_list),
            torch.stack(query_x_list), torch.tensor(query_y_list))

# --- 5. The Hybrid Prototypical Loss Function ---

def prototypical_loss_hybrid(encoder, attention_module, support_x, support_y, query_x, query_y, params: TrainingParams):
    """Computes loss and accuracy for one episode. Now also returns predictions."""
    support_embeddings = encoder(support_x)
    query_embeddings = encoder(query_x)
    support_embeddings = F.normalize(support_embeddings, p=2, dim=1)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    classes = torch.unique(support_y)
    all_dists = torch.zeros(len(query_y), len(classes)).to(DEVICE)
    class_sub_prototypes = {}
    for cls in classes:
        class_embs = support_embeddings[support_y == cls]
        class_sub_prototypes[cls.item()] = get_cluster_prototypes(class_embs, params.n_clusters_per_class)
    for i, query_emb in enumerate(query_embeddings):
        for j, cls in enumerate(classes):
            sub_prototypes = class_sub_prototypes[cls.item()]
            if sub_prototypes.nelement() == 0: continue
            attention_weights = attention_module(query_emb, sub_prototypes)
            final_prototype = torch.sum(attention_weights.unsqueeze(1) * sub_prototypes, dim=0)
            dist = torch.norm(query_emb - final_prototype, p=2)
            all_dists[i, j] = dist
    temperature = 2.0 
    log_p_y = F.log_softmax(-all_dists / temperature, dim=1)
    
    # Get predictions in terms of actual class labels
    preds_indices = log_p_y.argmax(dim=1)
    preds_labels = classes[preds_indices]
    
    query_labels_mapped = torch.tensor([torch.where(classes == y)[0].item() for y in query_y]).to(DEVICE)
    loss = F.nll_loss(log_p_y, query_labels_mapped)
    acc = (preds_indices == query_labels_mapped).float().mean()
    return loss, acc, preds_labels


# --- 6. Training and Evaluation Pipeline ---

def run_model(dataset, params: TrainingParams, mode='train'):
    if mode == 'train':
        encoder.train()
        attention_module.train()
    else:
        encoder.eval()
        attention_module.eval()
    total_loss, total_acc = 0.0, 0.0
    num_episodes = params.n_episodes if mode == 'train' else params.eval_every
    pbar_desc = f"{mode.capitalize()} Phase"
    pbar = tqdm(range(num_episodes), desc=pbar_desc)
    for episode in pbar:
        try:
            support_x, support_y, query_x, query_y = create_episode(
                dataset, params.n_way, params.k_shot, params.q_query
            )
        except ValueError as e:
            print(f"Skipping episode due to error: {e}")
            continue
        support_x, support_y = support_x.to(DEVICE), support_y.to(DEVICE)
        query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
        with torch.set_grad_enabled(mode == 'train'):
            loss, acc = prototypical_loss_hybrid(
                encoder, attention_module, support_x, support_y, query_x, query_y, params
            )
        total_loss += loss.item()
        total_acc += acc.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")
    return total_loss / num_episodes, total_acc / num_episodes

def evaluate_and_log_model(encoder, attention_module, test_set, params, n_test_episodes=1000):
    """
    Evaluates the model on test episodes and logs detailed results with statistics.
    """
    print("\n--- Starting Episodic Evaluation ---")
    encoder.eval()
    attention_module.eval()
    
    all_preds, all_labels = [], []
    episode_accuracies = []  # New: Track accuracy for each episode
    
    with torch.no_grad():
        for episode in tqdm(range(n_test_episodes), desc="Test Episodes"):
            try:
                support_x, support_y, query_x, query_y = create_episode(
                    test_set, params.n_way, params.k_shot, params.q_query
                )
            except ValueError as e:
                continue
            
            support_x, support_y = support_x.to(DEVICE), support_y.to(DEVICE)
            query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
            
            # Get support embeddings and create prototypes
            support_embeddings = encoder(support_x)
            support_embeddings = F.normalize(support_embeddings, p=2, dim=1)
            
            prototypes = {}
            for class_idx in torch.unique(support_y):
                class_mask = (support_y == class_idx)
                class_embeddings = support_embeddings[class_mask]
                class_prototypes = get_cluster_prototypes(class_embeddings, params.n_clusters_per_class)
                prototypes[class_idx.item()] = class_prototypes
            
            # Get query embeddings and predict
            query_embeddings = encoder(query_x)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            
            episode_preds = []
            for query_emb in query_embeddings:
                min_dist = float('inf')
                pred_class = None
                
                for class_idx, class_prototypes in prototypes.items():
                    if class_prototypes.nelement() == 0:
                        continue
                    
                    attention_weights = attention_module(query_emb, class_prototypes)
                    final_prototype = torch.sum(attention_weights.unsqueeze(1) * class_prototypes, dim=0)
                    dist = torch.norm(query_emb - final_prototype, p=2).item()
                    
                    if dist < min_dist:
                        min_dist = dist
                        pred_class = class_idx
                
                episode_preds.append(pred_class)
            
            # Calculate episode accuracy
            episode_correct = sum(1 for pred, true in zip(episode_preds, query_y.cpu().numpy()) if pred == true)
            episode_accuracy = episode_correct / len(query_y)
            episode_accuracies.append(episode_accuracy)
            
            all_preds.extend(episode_preds)
            all_labels.extend(query_y.cpu().numpy())
    
    # Calculate overall statistics
    overall_accuracy = sum(1 for pred, true in zip(all_preds, all_labels) if pred == true) / len(all_labels)
    
    # Calculate mean and standard deviation of episode accuracies
    mean_accuracy = np.mean(episode_accuracies) * 100
    std_accuracy = np.std(episode_accuracies) * 100
    
    print("\n--- Episodic Evaluation Results ---")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"Episodes Evaluated: {len(episode_accuracies)}")
    
    # Generate detailed classification report
    report_str = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0)
    print("\nDetailed Classification Report:")
    print(report_str)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # Save results
    report_path = "episodic_evaluation_report.txt"
    cm_path = "episodic_confusion_matrix.csv"
    
    with open(report_path, "w") as f:
        f.write("Episodic Evaluation Report\n")
        f.write("=========================\n\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%\n")
        f.write(f"Episodes Evaluated: {len(episode_accuracies)}\n")
        f.write(f"Test Configuration: {params.n_way}-way {params.k_shot}-shot with {params.q_query} queries\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    
    cm_df.to_csv(cm_path)
    
    print(f"\nEpisodic evaluation report saved to {report_path}")
    print(f"Episodic confusion matrix saved to {cm_path}")
    
    return overall_accuracy, all_preds, all_labels

def evaluate_with_full_prototypes(encoder, attention_module, train_set, val_set, test_set, params):
    """
    Evaluates the model by computing full prototypes from train+val sets and predicting on the entire test set.
    This approach uses all available training data to build robust class prototypes and tests on the complete test set.
    """
    print("\n--- Starting Full Test Set Evaluation ---")
    encoder.eval()
    attention_module.eval()
    
    # Step 1: Collect all embeddings from training and validation sets
    print("Computing embeddings for training and validation sets...")
    all_train_embeddings = {}
    all_val_embeddings = {}
    
    # Helper function to collect embeddings from a dataset
    def collect_embeddings(dataset, embedding_dict, dataset_name):
        if isinstance(dataset, torch.utils.data.Subset):
            subset_indices = dataset.indices
            original_dataset = dataset.dataset
        else:
            subset_indices = list(range(len(dataset)))
            original_dataset = dataset
            
        print(f"Processing {len(subset_indices)} samples from {dataset_name} set...")
        
        with torch.no_grad():
            for idx in tqdm(subset_indices, desc=f"Embedding {dataset_name}"):
                waveform, label = original_dataset[idx]
                waveform = waveform.unsqueeze(0).to(DEVICE)  # Add batch dimension
                
                embedding = encoder(waveform)
                embedding = F.normalize(embedding, p=2, dim=1).squeeze(0)  # Remove batch dimension
                
                if label not in embedding_dict:
                    embedding_dict[label] = []
                embedding_dict[label].append(embedding)
    
    # Collect embeddings from both train and validation sets
    collect_embeddings(train_set, all_train_embeddings, "training")
    collect_embeddings(val_set, all_val_embeddings, "validation")
    
    # Step 2: Combine train and val embeddings and compute full prototypes
    print("Computing full prototypes from combined train+val embeddings...")
    full_prototypes = {}
    
    for class_label in BREATHING_LABELS.values():
        train_embs = all_train_embeddings.get(class_label, [])
        val_embs = all_val_embeddings.get(class_label, [])
        
        # Combine train and validation embeddings
        combined_embs = train_embs + val_embs
        
        if combined_embs:
            combined_tensor = torch.stack(combined_embs)
            # Use the same clustering approach as in training
            class_prototypes = get_cluster_prototypes(combined_tensor, params.n_clusters_per_class)
            full_prototypes[class_label] = class_prototypes
            print(f"Class {class_label} ({LABEL_NAMES[class_label]}): {len(combined_embs)} samples -> {len(class_prototypes)} prototypes")
        else:
            print(f"Warning: No samples found for class {class_label} ({LABEL_NAMES[class_label]})")
    
    # Step 3: Process entire test set at once
    print("Processing entire test set...")
    if isinstance(test_set, torch.utils.data.Subset):
        test_indices = test_set.indices
        original_test_dataset = test_set.dataset
    else:
        test_indices = list(range(len(test_set)))
        original_test_dataset = test_set
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx in tqdm(test_indices, desc="Testing samples"):
            waveform, true_label = original_test_dataset[idx]
            waveform = waveform.unsqueeze(0).to(DEVICE)  # Add batch dimension
            
            # Get test sample embedding
            test_embedding = encoder(waveform)
            test_embedding = F.normalize(test_embedding, p=2, dim=1).squeeze(0)  # Remove batch dimension
            
            # Predict using full prototypes
            min_dist = float('inf')
            pred_class = None
            
            for class_label, class_prototypes in full_prototypes.items():
                if class_prototypes.nelement() == 0:
                    continue
                
                # Use attention mechanism to combine prototypes
                attention_weights = attention_module(test_embedding, class_prototypes)
                final_prototype = torch.sum(attention_weights.unsqueeze(1) * class_prototypes, dim=0)
                
                # Calculate distance
                dist = torch.norm(test_embedding - final_prototype, p=2).item()
                
                if dist < min_dist:
                    min_dist = dist
                    pred_class = class_label
            
            all_preds.append(pred_class)
            all_labels.append(true_label)
    
    # Step 4: Generate evaluation metrics
    print("\n--- Full Test Set Evaluation Results ---")
    
    # Calculate accuracy
    correct_predictions = sum(1 for pred, true in zip(all_preds, all_labels) if pred == true)
    accuracy = correct_predictions / len(all_labels)
    print(f"Overall Accuracy: {accuracy:.4f} ({correct_predictions}/{len(all_labels)})")
    
    # Generate detailed classification report
    report_str = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0)
    print("\nDetailed Classification Report:")
    print(report_str)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # Step 5: Save results
    full_test_report_path = "full_test_set_evaluation_report.txt"
    full_test_cm_path = "full_test_set_confusion_matrix.csv"
    
    with open(full_test_report_path, "w") as f:
        f.write("Full Test Set Evaluation Report\n")
        f.write("===============================\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Total Test Samples: {len(all_labels)}\n")
        f.write(f"Correct Predictions: {correct_predictions}\n\n")
        f.write("Methodology:\n")
        f.write("- Combined training and validation sets to compute robust class prototypes\n")
        f.write(f"- Used {params.n_clusters_per_class} sub-prototypes per class with attention weighting\n")
        f.write("- Evaluated on entire test set (not episodic)\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    
    cm_df.to_csv(full_test_cm_path)
    
    print(f"\nFull test set evaluation report saved to {full_test_report_path}")
    print(f"Full test set confusion matrix saved to {full_test_cm_path}")
    
    return accuracy, all_preds, all_labels


# --- 7. Main Execution Block ---
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    params = TrainingParams()

    AUDIO_DIR = "./audio"
    LABEL_DIR = "./strong_labels"
    if not os.path.isdir(AUDIO_DIR) or not os.path.isdir(LABEL_DIR):
        print("="*50 + "\n!!! WARNING: Audio or Label directory not found. !!!\n" +
              f"Please update AUDIO_DIR and LABEL_DIR in the script.\n" + "="*50)
        exit()

    
    # Create dataset with augmentation
    full_dataset = BreathingSegmentDataset(
        audio_dir=AUDIO_DIR, 
        label_dir=LABEL_DIR,
        augment_minority_classes=True,
        target_min_samples=100
    )
    
    # Split into train/val/test
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])
    print(f"\nData split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    # Initialize models
    encoder = AttentionEncoder(out_dim=params.embedding_dim).to(DEVICE)
    attention_module = QueryAttention(input_dim=params.embedding_dim).to(DEVICE)

    # Final Evaluation
    if os.path.exists(params.best_model_path):
        checkpoint = torch.load(params.best_model_path, weights_only=True)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        attention_module.load_state_dict(checkpoint['attention_module_state_dict'])
        print("\nLoaded best model for final evaluation.")
        evaluate_and_log_model(encoder, attention_module, test_set, params)
        # New full prototype evaluation
        print("\n" + "="*60)
        print("FULL PROTOTYPE EVALUATION (New Method)")
        print("="*60)
        evaluate_with_full_prototypes(encoder, attention_module, train_set, val_set, test_set, params)
    else:
        print("\nNo model was saved. Could not perform final evaluation.")
