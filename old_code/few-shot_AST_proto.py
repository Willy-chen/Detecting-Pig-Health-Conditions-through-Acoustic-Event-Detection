import os
import time
import glob
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from torch import nn
import torchaudio.transforms as T
from torch.nn import functional as F
from torch.utils.data import Dataset, random_split
from transformers import ASTFeatureExtractor, ASTModel

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

from pathlib import Path

# --- 1. Global Configuration & Hyperparameters ---

# Training Parameters
class TrainingParams:
    n_way = 5                # N: Number of classes in an episode
    max_samples = 20         # M: Maximum number of samples per class in an episode
    support_ratio = 0.7      # NEW: Ratio of data to use for support set
    n_episodes = 10       # Total training episodes
    eval_every = 10          # Evaluate on validation set every N episodes
    n_val_episodes = 10      # Number of validation episodes
    patience = 50           # Early stopping patience
    batch_size = 32          # Batch size for training

    sample_rate = 16000
    segment_length = 10 * sample_rate
    n_mels = 64              # Number of mel frequency bins
    n_fft = 1024             # FFT window size
    hop_length = 512         # Hop length for STFT

    embedding_dim = 64      # Final embedding dimension
    
    # Optimization (more conservative for transformers)
    learning_rate = 0.0001  # Lower learning rate
    weight_decay = 0.01     # Higher weight decay
    dropout_rate = 0.1      # Transformer dropout

    n_clusters_per_class = 5  # M: Number of k-means clusters (sub-prototypes)
    best_model_path = "best_hybrid_model.pth"
    train_log_path = "training_log.csv"
    eval_log_path = "validation_log.csv"
    final_report_path = "final_evaluation_report.txt"
    final_cm_path = "final_confusion_matrix.csv"

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
    def __init__(self, audio_dir, label_dir, params=None, 
                 augment_minority_classes=True, target_min_samples=50):
        self.data = []
        self.targets = []
        self.n_classes = len(BREATHING_LABELS)
        self.params = params if params else TrainingParams()
        self.augment_minority_classes = augment_minority_classes
        self.target_min_samples = target_min_samples
        
        print("Loading dataset...")
        self._load_original_data(audio_dir, label_dir)
        
        if self.augment_minority_classes:
            print("Augmenting minority classes...")
            self._augment_data()
            
        self._print_class_distribution()

    def _load_original_data(self, audio_dir, label_dir):
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
                if sr != self.params.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.params.sample_rate)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}, skipping...")
                continue

            num_segments = waveform.shape[1] // self.params.segment_length
            for i in range(num_segments):
                start = i * self.params.segment_length
                segment = waveform[:, start:start+self.params.segment_length]
                label = is_breathing_segment(start/self.params.sample_rate, 
                                             (start+self.params.segment_length)/self.params.sample_rate, 
                                             labels)
                
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
        def change_volume(waveform, factor):
            """Change volume by a factor."""
            return torch.clamp(waveform * factor, -1.0, 1.0)
        
        def pitch_shift_simple(waveform, shift_steps=2):
            """Simple pitch shift using resampling."""
            try:
                # Use more stable pitch shifting parameters
                shifted = torchaudio.transforms.PitchShift(
                    sample_rate=self.params.sample_rate, 
                    n_steps=shift_steps
                )(waveform)
                return shifted
            except Exception as e:
                # Log the error instead of silently falling back
                print(f"Pitch shift failed: {e}")
                return waveform
        
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
        return waveform.clone().detach(), label

# --- 3. Model Components ---
class ASTEncoder(nn.Module):
    def __init__(self, out_dim=64, sample_rate=16000, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.out_dim = out_dim
        
        # Initialize AST feature extractor and model
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.ast_model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        
        # Move AST model to device and freeze parameters
        self.ast_model.to(self.device)
        self.ast_model.eval()
        for param in self.ast_model.parameters():
            param.requires_grad = False
        
        # Get AST output dimension (typically 768 for base model)
        ast_output_dim = self.ast_model.config.hidden_size
        
        # Attention mechanism for weighted averaging of patches
        self.attention = nn.Sequential(
            nn.Linear(ast_output_dim, ast_output_dim // 2),
            nn.Tanh(),
            nn.Linear(ast_output_dim // 2, 1)
        )
        self.attention.to(self.device)
        
        # Projection layer to desired output dimension
        self.proj = nn.Linear(ast_output_dim, out_dim)
        self.proj.to(self.device)
    
    def _process_batch(self, audio_batch):
        """Process batch audio waveform to AST features."""
        batch_size = audio_batch.size(0)
        
        # Convert to numpy for feature extraction
        audio_np_batch = audio_batch.detach().cpu().numpy()
        
        batch_features = []
        
        for i in range(batch_size):
            try:
                # Use AST feature extractor to convert waveform to spectrogram
                inputs = self.feature_extractor(
                    audio_np_batch[i], 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                )
                
                # Move inputs to device
                input_values = inputs['input_values'].to(self.device)
                
                # Get AST features
                with torch.no_grad():
                    outputs = self.ast_model(input_values)
                    # outputs.last_hidden_state shape: [1, num_patches + 1, hidden_size]
                    # We exclude the CLS token (first token) and use patch tokens
                    patch_features = outputs.last_hidden_state[0, 1:, :]  # [num_patches, hidden_size]
                
                batch_features.append(patch_features)
                
            except Exception as e:
                print(f"Error processing batch item {i}: {e}")
                # Create dummy features for failed processing
                dummy_features = torch.zeros((100, self.ast_model.config.hidden_size)).to(self.device)
                batch_features.append(dummy_features)
        
        return batch_features
    
    def forward(self, x):
        """
        Forward pass with batch processing.
        Args:
            x: Input tensor of shape [B, 1, T] or [B, T] where B is batch size, T is time
        Returns:
            Projected AST features of shape [B, out_dim]
        """
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Process batch through AST
        ast_features_batch = self._process_batch(x)
        
        # Apply trainable attention and projection
        final_features = []
        for features in ast_features_batch:
            # Trainable attention mechanism over patch features
            attention_scores = self.attention(features).squeeze(-1)  # [num_patches]
            attention_weights = torch.softmax(attention_scores, dim=0)  # [num_patches]
            
            # Weighted averaging (gradients flow through attention weights)
            weighted_features = torch.sum(
                attention_weights.unsqueeze(1) * features, dim=0
            )  # [hidden_size]
            
            final_features.append(weighted_features)
        
        # Trainable projection
        batch_features = torch.stack(final_features)  # [B, hidden_size]
        return self.proj(batch_features)  # [B, out_dim]
    
    def to(self, device):
        """Override to method to ensure all components move to device."""
        super().to(device)
        self.device = device
        self.ast_model.to(device)
        self.attention.to(device)
        self.proj.to(device)
        return self

class PrototypicalClassifier(nn.Module):
    def __init__(self, feature_dim=64, num_classes=5, n_clusters_per_class=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.n_clusters_per_class = n_clusters_per_class

        self.query_attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
    def compute_prototypes(self, support_features, support_labels):
        """
        Compute class prototypes using K-means clustering.
        Args:
            support_features: [N, feature_dim] - features from support samples
            support_labels: [N] - labels for support samples
        Returns:
            class_prototypes: dict {class_idx: [k_clusters, feature_dim]}
        """
        class_prototypes = {}
        
        for class_idx in range(self.num_classes):
            # Find samples belonging to this class
            class_mask = (support_labels == class_idx)
            if class_mask.sum() > 0:
                class_features = support_features[class_mask]
                
                # Use K-means to get sub-prototypes
                cluster_prototypes = get_cluster_prototypes(
                    class_features, 
                    self.n_clusters_per_class
                )
                class_prototypes[class_idx] = cluster_prototypes
            else:
                # Handle case where no samples exist for this class
                dummy_prototypes = torch.zeros(
                    1, self.feature_dim, 
                    device=support_features.device
                )
                class_prototypes[class_idx] = dummy_prototypes
        
        return class_prototypes

    def compute_query_attention_weights(self, query_feature, class_prototypes):
        """
        Compute attention weights between a query and class prototypes.
        Args:
            query_feature: [feature_dim] - single query feature
            class_prototypes: [k_clusters, feature_dim] - prototypes for one class
        Returns:
            attention_weights: [k_clusters] - normalized attention weights
        """
        # Expand query to match prototype dimensions
        query_expanded = query_feature.unsqueeze(0).expand(
            class_prototypes.size(0), -1
        )  # [k_clusters, feature_dim]
        
        # Concatenate query and prototypes for attention computation
        combined_features = torch.cat([
            query_expanded, class_prototypes
        ], dim=1)  # [k_clusters, feature_dim * 2]
        
        # Compute attention scores
        attention_scores = self.query_attention(combined_features).squeeze(-1)  # [k_clusters]
        
        # Apply softmax to get normalized weights
        attention_weights = torch.softmax(attention_scores, dim=0)
        
        return attention_weights
    
    def compute_weighted_prototypes(self, query_features, class_prototypes_dict):
        """
        Compute weighted prototypes for each query using attention.
        Args:
            query_features: [M, feature_dim] - query sample features
            class_prototypes_dict: dict {class_idx: [k_clusters, feature_dim]}
        Returns:
            weighted_prototypes: [M, num_classes, feature_dim] - weighted prototypes for each query
        """
        batch_size = query_features.size(0)
        weighted_prototypes = torch.zeros(
            batch_size, self.num_classes, self.feature_dim,
            device=query_features.device
        )
        
        for query_idx in range(batch_size):
            query_feature = query_features[query_idx]  # [feature_dim]
            
            for class_idx, class_prototypes in class_prototypes_dict.items():
                if class_prototypes.size(0) > 0:
                    # Compute attention weights for this query-class pair
                    attention_weights = self.compute_query_attention_weights(
                        query_feature, class_prototypes
                    )  # [k_clusters]
                    
                    # Compute weighted sum of prototypes
                    weighted_prototype = torch.sum(
                        attention_weights.unsqueeze(1) * class_prototypes, 
                        dim=0
                    )  # [feature_dim]
                    
                    weighted_prototypes[query_idx, class_idx] = weighted_prototype
                else:
                    # Use zero vector if no prototypes available
                    weighted_prototypes[query_idx, class_idx] = torch.zeros(
                        self.feature_dim, device=query_features.device
                    )
        
        return weighted_prototypes
    
    def compute_distances_to_weighted_prototypes(self, query_features, weighted_prototypes):
        """
        Compute distances between queries and their weighted prototypes.
        Args:
            query_features: [M, feature_dim] - query sample features
            weighted_prototypes: [M, num_classes, feature_dim] - weighted prototypes
        Returns:
            distances: [M, num_classes] - distances to weighted prototypes
        """
        # Expand query features for broadcasting
        query_expanded = query_features.unsqueeze(1)  # [M, 1, feature_dim]
        
        # Compute Euclidean distances
        distances = torch.norm(
            query_expanded - weighted_prototypes, 
            p=2, dim=2
        )  # [M, num_classes]
        
        return distances
    
    def forward(self, query_features, support_features=None, support_labels=None, 
                class_prototypes_dict=None):
        """
        Enhanced forward pass with K-means clustering and query attention.
        Args:
            query_features: [M, feature_dim] - query sample features
            support_features: [N, feature_dim] - support sample features (optional)
            support_labels: [N] - support sample labels (optional)
            class_prototypes_dict: precomputed prototypes (optional)
        Returns:
            logits: [M, num_classes] - classification logits
        """
        if class_prototypes_dict is None:
            if support_features is None or support_labels is None:
                raise ValueError("Either class_prototypes_dict or support set must be provided")
            class_prototypes_dict = self.compute_prototypes(
                support_features, support_labels
            )
        
        # Compute query-specific weighted prototypes
        weighted_prototypes = self.compute_weighted_prototypes(
            query_features, class_prototypes_dict
        )
        
        # Compute distances to weighted prototypes
        distances = self.compute_distances_to_weighted_prototypes(
            query_features, weighted_prototypes
        )
        
        # Convert distances to logits (negative distances for softmax)
        logits = -distances
        
        return logits

# --- 4. Training Functions ---
def train(encoder, classifier, train_set, val_set, n_episodes, 
          optimizer, device, n_way, k_shot, q_query, params):
    """
    Train prototypical network with episodic learning.
    """
    start_time = time.time()
    train_history = {"episode": [], "loss": [], "accuracy": []}
    eval_history = {"episode": [], "loss": [], "accuracy": []}
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Convert datasets to lists for episode sampling
    train_data = [(train_set[i][0], train_set[i][1]) for i in range(len(train_set))]
    val_data = [(val_set[i][0], val_set[i][1]) for i in range(len(val_set))]
    
    print(f"Starting training for {n_episodes} episodes...")
    print(f"Episode config: {n_way}-way, {k_shot}-shot, {q_query}-query")
    
    pbar = tqdm(range(n_episodes), desc="Training Episodes")
    for episode in pbar:
        # Sample training episode
        episode_data = sample_episode(train_data, n_way, k_shot, q_query)
        
        # Move data to device
        support_audio = episode_data['support_audio'].to(device)
        support_labels = episode_data['support_labels'].to(device)
        query_audio = episode_data['query_audio'].to(device)
        query_labels = episode_data['query_labels'].to(device)
        
        # Training step
        encoder.train()
        classifier.train()
        
        # Extract features
        support_features = encoder(support_audio)
        query_features = encoder(query_audio)
        
        # Compute prototypes and classify
        logits = classifier(query_features, support_features, support_labels)
        
        # Compute loss
        loss = F.cross_entropy(logits, query_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        pred = logits.argmax(dim=1)
        accuracy = (pred == query_labels).float().mean()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy.item():.4f}'
        })
        train_history["episode"].append(episode)
        train_history["loss"].append(loss.item())
        train_history["accuracy"].append(accuracy.item())

        # Validation every eval_every episodes
        if (episode + 1) % params.eval_every == 0:
            val_acc, val_loss = validate_prototypical_network(
                encoder, classifier, val_data, device, n_way, k_shot, q_query, params
            )
            
            print(f"Episode {episode+1}/{n_episodes}")
            print(f"  Train Loss: {loss.item():.4f}, Train Acc: {accuracy.item():.4f}")
            print(f"  Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            eval_history["episode"].append(episode + 1)
            eval_history["loss"].append(val_loss)
            eval_history["accuracy"].append(val_acc)
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode,
                    'best_val_acc': best_val_acc
                }, params.best_model_path)
                
                print(f"  New best model saved! Val Acc: {val_acc:.4f}")
            else:
                patience_counter += params.eval_every
                
            if patience_counter >= params.patience:
                print(f"Early stopping at episode {episode+1}")
                break
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.2f} minutes.")
    with open('train_log.txt', 'w') as f:
        f.write(f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")
    
    pd.DataFrame(train_history).to_csv(params.train_log_path, index=False)
    print(f"Training history saved to {params.train_log_path}")
    pd.DataFrame(eval_history).to_csv(params.eval_log_path, index=False)
    print(f"Validation history saved to {params.eval_log_path}")
    return best_val_acc

def sample_episode(dataset, n_way, k_shot, q_query):
    """
    Sample an episode from the dataset for prototypical learning.
    Args:
        dataset: List of (audio, label) tuples
        n_way: Number of classes per episode
        k_shot: Number of support samples per class
        q_query: Number of query samples per class
    Returns:
        Dictionary with support and query sets
    """
    # Group data by class
    class_data = {}
    for audio, label in dataset:
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(audio)
    
    # Ensure we have enough classes
    available_classes = list(class_data.keys())
    if len(available_classes) < n_way:
        # If not enough classes, use all available classes
        selected_classes = available_classes
        n_way = len(available_classes)
    else:
        # Randomly select n_way classes
        selected_classes = random.sample(available_classes, n_way)
    
    support_audio = []
    support_labels = []
    query_audio = []
    query_labels = []
    
    for class_idx, original_class in enumerate(selected_classes):
        class_samples = class_data[original_class]
        
        # Ensure we have enough samples
        total_needed = k_shot + q_query
        if len(class_samples) < total_needed:
            # If not enough samples, use all available and duplicate if necessary
            available_samples = class_samples
            while len(available_samples) < total_needed:
                available_samples.extend(class_samples)
            class_samples = available_samples[:total_needed]
        
        # Randomly sample support and query sets
        sampled_indices = random.sample(range(len(class_samples)), total_needed)
        support_indices = sampled_indices[:k_shot]
        query_indices = sampled_indices[k_shot:k_shot + q_query]
        
        # Add to support set
        for idx in support_indices:
            support_audio.append(class_samples[idx])
            support_labels.append(class_idx)  # Use episode-specific class index
        
        # Add to query set
        for idx in query_indices:
            query_audio.append(class_samples[idx])
            query_labels.append(class_idx)  # Use episode-specific class index
    
    return {
        'support_audio': torch.stack(support_audio),
        'support_labels': torch.tensor(support_labels, dtype=torch.long),
        'query_audio': torch.stack(query_audio),
        'query_labels': torch.tensor(query_labels, dtype=torch.long)
    }

def validate_prototypical_network(encoder, classifier, val_data, device, n_way, k_shot, q_query, params):
    """
    Validate the prototypical network.
    """
    encoder.eval()
    classifier.eval()
    
    val_accuracies, val_losses = [], []
    n_val_episodes = params.n_val_episodes  # Number of validation episodes
    
    with torch.no_grad():
        for _ in range(n_val_episodes):
            # Sample validation episode
            episode_data = sample_episode(val_data, n_way, k_shot, q_query)
            
            support_audio = episode_data['support_audio'].to(device)
            support_labels = episode_data['support_labels'].to(device)
            query_audio = episode_data['query_audio'].to(device)
            query_labels = episode_data['query_labels'].to(device)
            
            # Extract features
            support_features = encoder(support_audio)
            query_features = encoder(query_audio)
            
            # Classify
            logits = classifier(query_features, support_features, support_labels)
            loss = F.cross_entropy(logits, query_labels)
            pred = logits.argmax(dim=1)
            accuracy = (pred == query_labels).float().mean()
            val_accuracies.append(accuracy.item())
            val_losses.append(loss.item())
    
    return np.mean(val_accuracies), np.mean(val_losses)

# --- 5. Helper Functions for Evaluation ---
def create_episode_full_data(dataset, n_way, support_ratio=0.8, max_samples=50):
    """
    Create episode data from full dataset for evaluation.
    """
    # Group data by class
    class_data = {}
    if isinstance(dataset, torch.utils.data.Subset):
        subset_indices = dataset.indices
        original_dataset = dataset.dataset
    else:
        subset_indices = list(range(len(dataset)))
        original_dataset = dataset
    
    for idx in subset_indices:
        waveform, label = original_dataset[idx]
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(waveform)
    
    # Select classes and samples
    available_classes = list(class_data.keys())
    selected_classes = available_classes[:n_way] if len(available_classes) >= n_way else available_classes
    
    support_audio = []
    support_labels = []
    query_audio = []
    query_labels = []
    
    for class_idx, original_class in enumerate(selected_classes):
        class_samples = class_data[original_class]
        
        # Limit samples per class
        if len(class_samples) > max_samples:
            class_samples = random.sample(class_samples, max_samples)
        
        # Split into support and query
        n_support = max(1, int(len(class_samples) * support_ratio))
        n_query = len(class_samples) - n_support
        
        # Randomly sample support and query sets
        sampled_indices = random.sample(range(len(class_samples)), len(class_samples))
        support_indices = sampled_indices[:n_support]
        query_indices = sampled_indices[n_support:n_support + n_query]
        
        # Add to support set
        for idx in support_indices:
            support_audio.append(class_samples[idx])
            support_labels.append(original_class)  # Use original class labels
        
        # Add to query set
        for idx in query_indices:
            query_audio.append(class_samples[idx])
            query_labels.append(original_class)  # Use original class labels
    
    return (torch.stack(support_audio), torch.tensor(support_labels, dtype=torch.long),
            torch.stack(query_audio), torch.tensor(query_labels, dtype=torch.long))

def get_cluster_prototypes(embeddings, n_clusters):
    """
    Get cluster prototypes using K-means.
    """
    if len(embeddings) <= n_clusters:
        print(f"Warning: Not enough embeddings ({len(embeddings)}) for {n_clusters} clusters. Returning original embeddings.")
        return embeddings
    
    # Convert to numpy for sklearn
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings_np)
    
    # Convert centroids back to tensors
    centroids = torch.from_numpy(kmeans.cluster_centers_).to(embeddings.device)
    return centroids

# --- 6. Evaluation Functions ---
def evaluate_and_log_model(encoder, classifier, test_set, params):
    """
    Evaluates the model on test episodes and logs detailed results with statistics.
    """
    print("\n--- Starting Episodic Evaluation ---")
    encoder.eval()
    classifier.eval()
    n_episode = params.n_val_episodes
    
    if isinstance(test_set, torch.utils.data.Subset):
        test_data = [(test_set.dataset[i][0], test_set.dataset[i][1]) for i in test_set.indices]
    else:
        test_data = [(test_set[i][0], test_set[i][1]) for i in range(len(test_set))]

    n_way = params.n_way
    k_shot = int(params.max_samples * params.support_ratio)
    q_query = int(params.max_samples * (1 - params.support_ratio))
    print(f"Episode configuration: {n_way}-way, {k_shot}-shot, {q_query}-query")

    accuracies = []
    losses = []
    all_predictions = []
    all_true_labels = []
    all_confidences = []

    with torch.no_grad():
        for episode in tqdm(range(n_episode), desc="Test Episodes"):
            try:
                # Sample episode from test data
                episode_data = sample_episode(test_data, n_way, k_shot, q_query)
                
                # Move data to device
                support_audio = episode_data['support_audio'].to(DEVICE)
                support_labels = episode_data['support_labels'].to(DEVICE)
                query_audio = episode_data['query_audio'].to(DEVICE)
                query_labels = episode_data['query_labels'].to(DEVICE)
                
                # Extract features
                support_features = encoder(support_audio)
                query_features = encoder(query_audio)
                
                # Classify
                logits = classifier(query_features, support_features, support_labels)
                loss = F.cross_entropy(logits, query_labels)
                pred = logits.argmax(dim=1)
                accuracy = (pred == query_labels).float().mean()
                
                # Store results
                accuracies.append(accuracy.item())
                losses.append(loss.item())
                all_predictions.extend(pred.cpu().numpy())
                all_true_labels.extend(query_labels.cpu().numpy())
                
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue
    
    # Calculate statistics
    avg_accuracy = np.mean(accuracies) if accuracies else 0.0
    std_accuracy = np.std(accuracies) if accuracies else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    std_loss = np.std(losses) if losses else 0.0
    
    # Generate classification report
    if all_predictions and all_true_labels:
        unique_labels = sorted(set(all_true_labels))
        class_names = [LABEL_NAMES[i] if i < len(LABEL_NAMES) else f"Class_{i}" for i in unique_labels]
        
        classification_rep = classification_report(
            all_true_labels, all_predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Print results
    print(f"\nEpisodic Test Evaluation Results ({len(accuracies)} successful episodes):")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"Best Episode Accuracy: {max(accuracies):.4f}" if accuracies else "N/A")
    print(f"Worst Episode Accuracy: {min(accuracies):.4f}" if accuracies else "N/A")
    
    # Save detailed results
    episodic_report_path = "episodic_test_evaluation_report.txt"
    episodic_cm_path = "episodic_test_confusion_matrix.csv"
    
    with open(episodic_report_path, 'w') as f:
        f.write("Episodic Test Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of episodes: {n_episode}\n")
        f.write(f"Successful episodes: {len(accuracies)}\n")
        f.write(f"Episode config: {n_way}-way, {k_shot}-shot, {q_query}-query\n\n")
        
        f.write("Overall Performance:\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f.write(f"Average Loss: {avg_loss:.4f} ± {std_loss:.4f}\n")
        f.write(f"Best Episode Accuracy: {max(accuracies):.4f}\n" if accuracies else "N/A\n")
        f.write(f"Worst Episode Accuracy: {min(accuracies):.4f}\n" if accuracies else "N/A\n")
        
        if all_predictions and all_true_labels:
            f.write("\nPer-Class Performance:\n")
            for class_name in class_names:
                if class_name in classification_rep:
                    metrics = classification_rep[class_name]
                    f.write(f"Class {class_name}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {metrics['support']}\n\n")
    
    # Save confusion matrix if available
    if all_predictions and all_true_labels:
        cm_df.to_csv(episodic_cm_path)
        print(f"Confusion matrix saved to: {episodic_cm_path}")
    
    print(f"Detailed report saved to: {episodic_report_path}")
    
    return avg_accuracy, avg_loss, accuracies, losses, all_predictions, all_true_labels

def evaluate_with_full_prototypes(encoder, classifier, train_set, val_set, test_set, params):
    """Fixed version with consistent device handling."""
    print("\n--- Starting Full Test Set Evaluation ---")
    encoder.eval()
    classifier.eval()
    
    # Step 1: Collect all embeddings from training and validation sets
    print("Computing embeddings for training and validation sets...")
    all_train_embeddings = {}
    all_val_embeddings = {}
    
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
                waveform = waveform.unsqueeze(0).to(DEVICE)
                
                embedding = encoder(waveform)
                embedding = F.normalize(embedding, p=2, dim=1).squeeze(0)
                
                if label not in embedding_dict:
                    embedding_dict[label] = []
                embedding_dict[label].append(embedding)  # Keep on GPU
    
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
            combined_tensor = torch.stack(combined_embs)  # Already on GPU
            class_prototypes = get_cluster_prototypes(combined_tensor, params.n_clusters_per_class)
            full_prototypes[class_label] = class_prototypes
            print(f"Class {class_label} ({LABEL_NAMES[class_label]}): {len(combined_embs)} samples -> {len(class_prototypes)} prototypes")
        else:
            print(f"Warning: No samples found for class {class_label} ({LABEL_NAMES[class_label]})")
    
    # Step 3: Process entire test set
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
            waveform = waveform.unsqueeze(0).to(DEVICE)
            
            test_embedding = encoder(waveform)
            test_embedding = F.normalize(test_embedding, p=2, dim=1).squeeze(0)
            
            # Predict using full prototypes
            min_dist = float('inf')
            pred_class = None
            
            for class_label, class_prototypes in full_prototypes.items():
                if class_prototypes.nelement() == 0:
                    continue
                
                attention_weights = classifier.compute_query_attention_weights(test_embedding, class_prototypes)
                final_prototype = torch.sum(attention_weights.unsqueeze(1) * class_prototypes, dim=0)
                
                dist = torch.norm(test_embedding - final_prototype, p=2).item()
                
                if dist < min_dist:
                    min_dist = dist
                    pred_class = class_label
            
            all_preds.append(pred_class)
            all_labels.append(true_label)
    
    # Step 4: Generate evaluation metrics
    print("\n--- Full Test Set Evaluation Results ---")
    
    correct_predictions = sum(1 for pred, true in zip(all_preds, all_labels) if pred == true)
    accuracy = correct_predictions / len(all_labels)
    print(f"Overall Accuracy: {accuracy:.4f} ({correct_predictions}/{len(all_labels)})")
    
    # Generate detailed classification report with zero_division parameter
    report_str = classification_report(
        all_labels, all_preds, 
        target_names=LABEL_NAMES, 
        zero_division=0  # Fix the warning
    )
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
        target_min_samples=params.max_samples*5,
    )
    
    # Split into train/val/test
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])
    print(f"\nData split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    # Initialize models
    encoder = ASTEncoder(
        out_dim=params.embedding_dim,  
        sample_rate=params.sample_rate, 
        device=DEVICE
    )
    encoder.to(DEVICE)

    classifier = PrototypicalClassifier(
        feature_dim=params.embedding_dim, 
        num_classes=params.n_way
    )
    classifier.to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.learning_rate, 
        weight_decay=params.weight_decay
    )

    # Calculate episode parameters
    k_shot = int(params.max_samples * params.support_ratio)
    q_query = int(params.max_samples * (1 - params.support_ratio))
    
    print(f"Episode configuration:")
    print(f"  N-way: {params.n_way}")
    print(f"  K-shot: {k_shot}")
    print(f"  Q-query: {q_query}")
    
    # Train the model
    best_val_acc = train(
        encoder=encoder,
        classifier=classifier,
        train_set=train_set,
        val_set=val_set,
        n_episodes=params.n_episodes,
        optimizer=optimizer,
        device=DEVICE,
        n_way=params.n_way,
        k_shot=k_shot,
        q_query=q_query,
        params=params
    )
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(params.best_model_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    # Run both evaluation functions
    print("\n" + "="*50)
    print("RUNNING EVALUATIONS")
    print("="*50)
    
    # Evaluation 1: Episodic evaluation
    accuracy1, loss1, _, _, preds1, labels1 = evaluate_and_log_model(
        encoder, classifier, test_set, params
    )
    
    # Evaluation 2: Full prototype evaluation
    accuracy2, preds2, labels2 = evaluate_with_full_prototypes(
        encoder, classifier, train_set, val_set, test_set, params
    )
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Episodic Evaluation Accuracy: {accuracy1:.4f}")
    print(f"Full Prototype Evaluation Accuracy: {accuracy2:.4f}")
    print("="*50)
