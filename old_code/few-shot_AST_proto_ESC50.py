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
    n_episodes = 2000        # Total training episodes (Reduced from 2000)
    eval_every = 10          # Evaluate on validation set every N episodes
    n_val_episodes = 10      # Number of validation episodes
    patience = 100           # Early stopping patience
    batch_size = 32          # Batch size for training

    sample_rate = 16000
    segment_length = 10 * sample_rate
    n_mels = 64              # Number of mel frequency bins
    n_fft = 1024             # FFT window size
    hop_length = 512         # Hop length for STFT

    embedding_dim = 64       # Final embedding dimension
    num_layers_to_unfreeze = 0 # Number of AST layers to fine-tune (0 freezes all layers)
    
    # Optimization (more conservative for transformers)
    head_lr = 1e-4           # LR for the new attention head
    finetune_lr = 1e-5       # Smaller LR for the pre-trained AST layers
    weight_decay = 0.01     # Higher weight decay
    dropout_rate = 0.1      # Transformer dropout

    # Number of frequency and time masks to apply
    n_freq_masks = 2
    n_time_masks = 2
    # Max size of the masks
    freq_mask_param = 30
    time_mask_param = 40

    n_clusters_per_class = 5  # M: Number of k-means clusters (sub-prototypes)
    best_model_path = "best_hybrid_model.pth"
    train_log_path = "training_log.csv"
    eval_log_path = "validation_log.csv"
    final_report_path = "final_evaluation_report.txt"
    final_cm_path = "final_confusion_matrix.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
print("Using device:", DEVICE)

# ESC-50 has 50 classes, but we'll work with 5-way episodes
NUM_ESC50_CLASSES = 50

# --- 2. ESC-50 Dataset Class ---

class ESC50Dataset(Dataset):
    def __init__(self, audio_dir, csv_path, folds=[1,2,3,4,5], segment_length=None):
        self.audio_dir = audio_dir
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['fold'].isin(folds)]
        self.df = self.df.reset_index(drop=True)
        self.sample_rate = SAMPLE_RATE  # AST expects 16kHz mono
        self.segment_length = segment_length if segment_length else 10 * self.sample_rate
        
        print(f"ESC50Dataset initialized with {len(self.df)} samples from folds {folds}")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution in the dataset."""
        class_counts = self.df['target'].value_counts().sort_index()
        print(f"Class distribution across {len(class_counts)} classes:")
        for class_id, count in class_counts.items():
            category = self.df[self.df['target'] == class_id]['category'].iloc[0]
            print(f"  Class {class_id} ({category}): {count} samples")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row['filename'])
        
        try:
            waveform, sr = torchaudio.load(wav_path)
            # Resample to 16kHz mono if needed
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Pad or trim to fixed length
            if waveform.shape[1] < self.segment_length:
                # Pad with zeros
                padding = self.segment_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            elif waveform.shape[1] > self.segment_length:
                # Trim to segment length
                waveform = waveform[:, :self.segment_length]
            
            label = int(row['target'])
            return waveform, label
            
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            # Return dummy data
            dummy_waveform = torch.zeros(1, self.segment_length)
            return dummy_waveform, 0

    def __len__(self):
        return len(self.df)

# --- 3. Model Components ---
class SimpleAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(embed_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.bmm(attention_weights, v)
        output = self.out_proj(context)
        
        return output

class ASTEncoder(nn.Module):
    def __init__(self, out_dim=64, sample_rate=16000, num_layers_to_unfreeze=0, dropout_rate=0.1, device=None, params=None):
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
        self.num_layers_to_unfreeze = num_layers_to_unfreeze
        print(f"Freezing all AST layers, then unfreezing the top {num_layers_to_unfreeze} layers.")
        for param in self.ast_model.parameters():
            param.requires_grad = False
        
        if num_layers_to_unfreeze > 0:
            for i in range(len(self.ast_model.encoder.layer) - num_layers_to_unfreeze, len(self.ast_model.encoder.layer)):
                for param in self.ast_model.encoder.layer[i].parameters():
                    param.requires_grad = True
            for param in self.ast_model.layernorm.parameters():
                param.requires_grad = True

        ast_output_dim = self.ast_model.config.hidden_size # 768
        
        # --- NEW: Adaptive Pooling to reduce sequence length ---
        self.patch_pooler = nn.AdaptiveAvgPool1d(128)

        # --- Simple Attention Block ---
        self.simple_attention = SimpleAttention(
            embed_dim=ast_output_dim,
            dropout=dropout_rate
        )
        self.layer_norm1 = nn.LayerNorm(ast_output_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(ast_output_dim, ast_output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ast_output_dim * 2, ast_output_dim)
        )
        self.layer_norm2 = nn.LayerNorm(ast_output_dim)
        
        # Final projection layer
        self.proj = nn.Linear(ast_output_dim, out_dim)
        
        # --- Data Augmentation ---
        self.spec_augment = T.SpecAugment(
            n_freq_masks=params.n_freq_masks,
            n_time_masks=params.n_time_masks,
            freq_mask_param=params.freq_mask_param,
            time_mask_param=params.time_mask_param
        )
        self.spec_augment.to(self.device)
        
        # Move new layers to device
        self.to(self.device)
    
    def _process_batch(self, audio_batch):
        """Process batch audio waveform to AST features."""
        inputs = self.feature_extractor(
            audio_batch.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt")
        
        input_values = inputs['input_values'].to(self.device)
        
        if self.training and hasattr(self, 'spec_augment'):
            input_values = self.spec_augment(input_values)
        
        # The ast_model also processes the entire batch at once
        if self.num_layers_to_unfreeze == 0:
            with torch.no_grad():
                outputs = self.ast_model(input_values)
        # Otherwise, run it with gradient tracking enabled
        else:
            outputs = self.ast_model(input_values)
        patch_features = outputs.last_hidden_state[:, 1:, :] # [B, num_patches, hidden_size]

        return patch_features
    
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
        
        # Process batch through AST to get patch embeddings
        patch_embeddings = self._process_batch(x) # [B, num_patches, hidden_size]
        
        # --- NEW: Pool patches before attention to reduce computation ---
        # (B, num_patches, hidden_size) -> (B, hidden_size, num_patches)
        patch_embeddings_t = patch_embeddings.transpose(1, 2)
        # (B, hidden_size, num_patches) -> (B, hidden_size, pooled_patches)
        pooled_patch_embeddings_t = self.patch_pooler(patch_embeddings_t)
        # (B, hidden_size, pooled_patches) -> (B, pooled_patches, hidden_size)
        pooled_patch_embeddings = pooled_patch_embeddings_t.transpose(1, 2)

        # Apply the new transformer-style attention block on pooled patches
        # 1. Simple Attention
        attn_output = self.simple_attention(pooled_patch_embeddings)
        # 2. Add & Norm (skip connection)
        x = self.layer_norm1(pooled_patch_embeddings + attn_output)
        # 3. Feed Forward
        ff_output = self.feed_forward(x)
        # 4. Add & Norm (skip connection)
        x = self.layer_norm2(x + ff_output)
        
        # Pool features across the patch dimension
        pooled_output = x.mean(dim=1) # [B, hidden_size]
        
        # Final projection to desired embedding dimension
        return self.proj(pooled_output)
    
    def to(self, device):
        """Override to method to ensure all components move to device."""
        super().to(device)
        self.device = device
        self.ast_model.to(device)
        self.spec_augment.to(device)
        self.proj.to(device)
        return self

class PrototypicalClassifier(nn.Module):
    def __init__(self, feature_dim=64, num_classes=5, n_clusters_per_class=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.query_attention = SimpleAttention(embed_dim=feature_dim)
        
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
        Compute attention weights between a query and class prototypes using transformer-style attention.
        Args:
            query_feature: [feature_dim] - single query feature
            class_prototypes: [k_clusters, feature_dim] - prototypes for one class
        Returns:
            attention_weights: [k_clusters] - normalized attention weights
        """
        # query_feature: [feature_dim], class_prototypes: [k_clusters, feature_dim]
        # Prepare for attention: batch_size=1, seq_len=k_clusters+1, embed_dim=feature_dim
        # Concatenate query at start of prototypes
        concat = torch.cat([query_feature.unsqueeze(0), class_prototypes], dim=0).unsqueeze(0)  # [1, k_clusters+1, feature_dim]
        attn_out = self.query_attention(concat)  # [1, k_clusters+1, feature_dim]
        # Get attention weights from the query to each prototype
        # Compute dot-product attention weights manually (as in SimpleAttention)
        q = self.query_attention.query(query_feature.unsqueeze(0).unsqueeze(0))  # [1,1,feature_dim]
        k = self.query_attention.key(class_prototypes.unsqueeze(0))              # [1,k_clusters,feature_dim]
        scores = torch.bmm(q, k.transpose(1,2)) / math.sqrt(self.feature_dim)     # [1,1,k_clusters]
        attention_weights = torch.softmax(scores, dim=-1).squeeze(0).squeeze(0)   # [k_clusters]
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

# --- 4. Training and Helper Functions ---

def sample_episode(dataset, n_way, k_shot, q_query):
    """
    Sample an episode from the ESC-50 dataset for prototypical learning.
    Args:
        dataset: ESC50Dataset instance
        n_way: Number of classes per episode
        k_shot: Number of support samples per class
        q_query: Number of query samples per class
    Returns:
        Dictionary with support and query sets
    """
    # Group data by class
    class_data = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(idx)
    
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
        class_indices = class_data[original_class]
        
        # Ensure we have enough samples
        total_needed = k_shot + q_query
        if len(class_indices) < total_needed:
            # If not enough samples, use all available and duplicate if necessary
            available_indices = class_indices
            while len(available_indices) < total_needed:
                available_indices.extend(class_indices)
            class_indices = available_indices[:total_needed]
        
        # Randomly sample support and query sets
        sampled_indices = random.sample(class_indices, total_needed)
        support_indices = sampled_indices[:k_shot]
        query_indices = sampled_indices[k_shot:k_shot + q_query]
        
        # Add to support set
        for idx in support_indices:
            audio, _ = dataset[idx]
            support_audio.append(audio)
            support_labels.append(class_idx)  # Use episode-specific class index
        
        # Add to query set
        for idx in query_indices:
            audio, _ = dataset[idx]
            query_audio.append(audio)
            query_labels.append(class_idx)  # Use episode-specific class index
    
    return {
        'support_audio': torch.stack(support_audio),
        'support_labels': torch.tensor(support_labels, dtype=torch.long),
        'query_audio': torch.stack(query_audio),
        'query_labels': torch.tensor(query_labels, dtype=torch.long)
    }

def validate_prototypical_network(encoder, classifier, val_set, device, n_way, k_shot, q_query, params):
    """
    Validate the prototypical network using episodic evaluation.
    """
    encoder.eval()
    classifier.eval()
    
    val_accuracies, val_losses = [], []
    n_val_episodes = params.n_val_episodes  # Number of validation episodes
    
    with torch.no_grad():
        for _ in range(n_val_episodes):
            # Sample validation episode
            episode_data = sample_episode(val_set, n_way, k_shot, q_query)
            
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

def evaluate_full_prototype_prediction(encoder, classifier, train_set, test_set, params, device):
    """
    Evaluate the model by computing prototypes from the full training set and testing on the full test set.
    Args:
        encoder: The feature encoder model
        classifier: The prototypical classifier
        train_set: The full training dataset
        test_set: The full test dataset
        params: Training parameters
        device: Device to run the model on
    Returns:
        test_accuracy: Accuracy on the full test set
        test_loss: Loss on the full test set
        predictions: All predictions
        true_labels: All true labels
    """
    print("Starting full prototype prediction evaluation...")
    encoder.eval()
    classifier.eval()

    # Extract features and labels from the full training set
    train_features = []
    train_labels = []
    with torch.no_grad():
        for i in tqdm(range(len(train_set)), desc="Extracting train features"):
            audio, label = train_set[i]
            audio = audio.unsqueeze(0).to(device)  # Add batch dimension
            feature = encoder(audio)  # [1, feature_dim]
            train_features.append(feature.squeeze(0))
            train_labels.append(label)
    
    train_features = torch.stack(train_features)  # [N_train, feature_dim]
    train_labels = torch.tensor(train_labels, dtype=torch.long, device=device)  # [N_train]

    # Get unique classes and map them to continuous indices
    unique_classes = torch.unique(train_labels).cpu().numpy()
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
    num_classes = len(unique_classes)
    
    print(f"Found {num_classes} classes in training set: {unique_classes}")
    
    # Remap training labels to continuous indices
    train_labels_mapped = torch.tensor([class_mapping[label.item()] for label in train_labels], 
                                      dtype=torch.long, device=device)

    # Update classifier num_classes for full evaluation
    original_num_classes = classifier.num_classes
    classifier.num_classes = num_classes

    # Compute prototypes for all classes in training set
    class_prototypes = classifier.compute_prototypes(train_features, train_labels_mapped)

    # Extract features and labels from the full test set
    test_features = []
    test_labels = []
    test_labels_mapped = []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_set)), desc="Extracting test features"):
            audio, label = test_set[i]
            audio = audio.unsqueeze(0).to(device)
            feature = encoder(audio)
            test_features.append(feature.squeeze(0))
            test_labels.append(label)
            # Map test labels using the same mapping
            if label in class_mapping:
                test_labels_mapped.append(class_mapping[label])
            else:
                # Handle unseen classes in test set (shouldn't happen in proper splits)
                print(f"Warning: Test label {label} not seen in training set")
                test_labels_mapped.append(-1)  # Mark as invalid
    
    test_features = torch.stack(test_features)  # [N_test, feature_dim]
    test_labels_mapped = torch.tensor(test_labels_mapped, dtype=torch.long, device=device)
    
    # Filter out invalid labels
    valid_mask = test_labels_mapped >= 0
    if valid_mask.sum() < len(test_labels_mapped):
        print(f"Warning: {len(test_labels_mapped) - valid_mask.sum()} test samples have unseen labels")
        test_features = test_features[valid_mask]
        test_labels_mapped = test_labels_mapped[valid_mask]

    # Classify test features using full prototypes
    logits = classifier(test_features, class_prototypes_dict=class_prototypes)

    # Compute loss and accuracy
    loss = F.cross_entropy(logits, test_labels_mapped)
    pred = logits.argmax(dim=1)
    accuracy = (pred == test_labels_mapped).float().mean().item()

    # Convert predictions back to original class labels for reporting
    pred_original = [unique_classes[p.item()] for p in pred]
    true_original = [unique_classes[t.item()] for t in test_labels_mapped]

    print(f"Full prototype prediction evaluation results:")
    print(f"  Test Loss: {loss.item():.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Total test samples: {len(test_features)}")

    # Restore original classifier num_classes
    classifier.num_classes = original_num_classes

    return accuracy, loss.item(), pred_original, true_original

def evaluate_episodic_on_test(encoder, classifier, test_set, device, n_way, k_shot, q_query, params):
    """
    Evaluate the model on test episodes (episodic evaluation).
    """
    print(f"\nStarting episodic evaluation on test set...")
    encoder.eval()
    classifier.eval()
    
    n_episode = params.n_val_episodes
    
    print(f"Episode configuration: {n_way}-way, {k_shot}-shot, {q_query}-query")

    accuracies = []
    losses = []
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for episode in tqdm(range(n_episode), desc="Test Episodes"):
            try:
                # Sample episode from test data
                episode_data = sample_episode(test_set, n_way, k_shot, q_query)
                
                # Move data to device
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
    
    print(f"Episodic evaluation results ({len(accuracies)} successful episodes):")
    print(f"Average Accuracy: {avg_accuracy:.4f} \u00b1 {std_accuracy:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    return avg_accuracy, avg_loss, accuracies, losses, all_predictions, all_true_labels

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
    
    print(f"Starting training for {n_episodes} episodes...")
    print(f"Episode config: {n_way}-way, {k_shot}-shot, {q_query}-query")
    
    pbar = tqdm(range(n_episodes), desc="Training Episodes")
    for episode in pbar:
        # Sample training episode
        episode_data = sample_episode(train_set, n_way, k_shot, q_query)
        
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
                encoder, classifier, val_set, device, n_way, k_shot, q_query, params
            )
            
            print(f"Episode {episode+1}/{n_episodes}")
            print(f"  Train Loss: {loss.item():.4f}, Train Acc: {accuracy.item():.4f}")
            print(f"  Val Acc (episodic): {val_acc:.4f}, Val Loss: {val_loss:.4f}")
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
                    'best_val_acc': best_val_acc,
                    'best_val_loss': best_val_loss
                }, params.best_model_path)
                
                print(f"  New best model saved! Val Acc: {val_acc:.4f}, val Loss: {val_loss:.4f}")
            else:
                patience_counter += params.eval_every
                
            if patience_counter >= params.patience:
                print(f"Early stopping at episode {episode+1}")
                break
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.2f} minutes.")
    
    pd.DataFrame(train_history).to_csv(params.train_log_path, index=False)
    print(f"Training history saved to {params.train_log_path}")
    pd.DataFrame(eval_history).to_csv(params.eval_log_path, index=False)
    print(f"Validation history saved to {params.eval_log_path}")
    return best_val_acc

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

# --- 5. 5-Fold Cross Validation with Both Evaluation Methods ---

def run_5_fold_cross_validation_dual_evaluation():
    """
    Run 5-fold cross-validation with both episodic and full prototype evaluation methods.
    """
    # ESC-50 dataset paths
    AUDIO_DIR = "./ESC-50-master/audio"
    CSV_PATH = "./ESC-50-master/meta/esc50.csv"
    
    if not os.path.exists(AUDIO_DIR) or not os.path.exists(CSV_PATH):
        print("="*50 + "\n!!! WARNING: ESC-50 dataset not found. !!!\n" +
              f"Please ensure ESC-50 dataset is available at:\n" +
              f"Audio: {AUDIO_DIR}\n" +
              f"CSV: {CSV_PATH}\n" + "="*50)
        return
    
    params = TrainingParams()
    fold_results = []
    
    print("Starting 5-Fold Cross Validation on ESC-50 with Dual Evaluation Methods")
    print("="*80)
    print("Training: Episodic learning (5-way few-shot)")
    print("Validation: Episodic evaluation")
    print("Testing: Both episodic and full prototype evaluation")
    print("="*80)
    
    for fold in range(1, 6):
        print(f"\n{'='*25} FOLD {fold} {'='*25}")
        
        # Create datasets for current fold
        train_folds = [f for f in range(1, 6) if f != fold]
        val_fold = [fold]
        test_fold = [fold]
        
        print(f"Train folds: {train_folds}")
        print(f"Validation fold: {val_fold}")
        print(f"Test fold: {test_fold}")
        
        # Create datasets
        train_dataset = ESC50Dataset(AUDIO_DIR, CSV_PATH, folds=train_folds)
        val_dataset = ESC50Dataset(AUDIO_DIR, CSV_PATH, folds=val_fold)
        test_dataset = ESC50Dataset(AUDIO_DIR, CSV_PATH, folds=test_fold)
        
        print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Initialize models for this fold
        encoder = ASTEncoder(
            out_dim=params.embedding_dim,  
            sample_rate=params.sample_rate,
            num_layers_to_unfreeze=params.num_layers_to_unfreeze,
            dropout_rate=params.dropout_rate,
            device=DEVICE,
            params=params
        )
        encoder.to(DEVICE)

        classifier = PrototypicalClassifier(
            feature_dim=params.embedding_dim, 
            num_classes=params.n_way,
            n_clusters_per_class=params.n_clusters_per_class
        )
        classifier.to(DEVICE)

        print("Setting up optimizer with discriminative learning rates...")
        ast_params = [p for p in encoder.ast_model.parameters() if p.requires_grad]
        head_params = list(encoder.patch_pooler.parameters()) + \
                      list(encoder.simple_attention.parameters()) + \
                      list(encoder.layer_norm1.parameters()) + \
                      list(encoder.feed_forward.parameters()) + \
                      list(encoder.layer_norm2.parameters()) + \
                      list(encoder.proj.parameters()) + \
                      list(classifier.parameters())
        
        num_ast_params = sum(p.numel() for p in ast_params)
        num_head_params = sum(p.numel() for p in head_params)
        
        print(f"  - Fine-tuning {num_ast_params:,} parameters in AST.")
        print(f"  - Training {num_head_params:,} parameters in the new head.")

        optimizer = torch.optim.AdamW([
            {'params': ast_params, 'lr': params.finetune_lr},
            {'params': head_params, 'lr': params.head_lr}
        ], weight_decay=params.weight_decay)

        # Calculate episode parameters for training
        k_shot = int(params.max_samples * params.support_ratio)
        q_query = int(params.max_samples * (1 - params.support_ratio))
        
        # Update model paths for this fold
        params.best_model_path = f"best_model_fold_{fold}.pth"
        params.train_log_path = f"training_log_fold_{fold}.csv"
        params.eval_log_path = f"validation_log_fold_{fold}.csv"
        
        print(f"Training configuration for fold {fold}:")
        print(f"  N-way: {params.n_way}")
        print(f"  K-shot: {k_shot}")
        print(f"  Q-query: {q_query}")
        
        # Train the model for this fold using episodic learning
        print(f"\nTraining model for fold {fold} with episodic learning...")
        best_val_acc = train(
            encoder=encoder,
            classifier=classifier,
            train_set=train_dataset,
            val_set=val_dataset,
            n_episodes=params.n_episodes,
            optimizer=optimizer,
            device=DEVICE,
            n_way=params.n_way,
            k_shot=k_shot,
            q_query=q_query,
            params=params
        )
        
        # Load best model for evaluation
        print(f"\nLoading best model for fold {fold} evaluation...")
        checkpoint = torch.load(params.best_model_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        print(f"\n{'='*15} EVALUATION METHODS FOR FOLD {fold} {'='*15}")
        
        # Method 1: Episodic Evaluation on Test Set
        print(f"\n--- Method 1: Episodic Evaluation ---")
        episodic_test_acc, episodic_test_loss, _, _, _, _ = evaluate_episodic_on_test(
            encoder, classifier, test_dataset, DEVICE, params.n_way, k_shot, q_query, params
        )
        
        # Method 2: Full Prototype Evaluation
        print(f"\n--- Method 2: Full Prototype Evaluation ---")
        full_test_acc, full_test_loss, predictions, true_labels = evaluate_full_prototype_prediction(
            encoder, classifier, train_dataset, test_dataset, params, DEVICE
        )
        
        # Get class names for the report
        unique_test_classes = sorted(list(set(true_labels)))
        class_names = [f"Class_{c}" for c in unique_test_classes]
        
        # Generate detailed classification report
        report = classification_report(
            true_labels, predictions, 
            labels=unique_test_classes,
            target_names=class_names,
            zero_division=0
        )
        
        print(f"\nClassification Report for Fold {fold} (Full Prototype Method):")
        print(report)
        
        # Save classification report
        with open(f"classification_report_fold_{fold}.txt", "w") as f:
            f.write(f"Classification Report for Fold {fold} (Full Prototype Method)\n")
            f.write("="*60 + "\n")
            f.write(report)
        
        # Store fold results
        fold_result = {
            'fold': fold,
            'best_val_acc_episodic': best_val_acc,
            'test_acc_episodic': episodic_test_acc,
            'test_loss_episodic': episodic_test_loss,
            'test_acc_full_prototype': full_test_acc,
            'test_loss_full_prototype': full_test_loss,
            'num_test_samples_full': len(predictions)
        }
        fold_results.append(fold_result)
        
        print(f"\n{'='*15} FOLD {fold} SUMMARY {'='*15}")
        print(f"Best Validation Accuracy (episodic): {best_val_acc:.4f}")
        print(f"Test Accuracy (episodic): {episodic_test_acc:.4f}")
        print(f"Test Loss (episodic): {episodic_test_loss:.4f}")
        print(f"Test Accuracy (full prototype): {full_test_acc:.4f}")
        print(f"Test Loss (full prototype): {full_test_loss:.4f}")
        print(f"Test samples evaluated (full): {len(predictions)}")
    
    # Calculate overall statistics
    val_accs = [result['best_val_acc_episodic'] for result in fold_results]
    episodic_test_accs = [result['test_acc_episodic'] for result in fold_results]
    episodic_test_losses = [result['test_loss_episodic'] for result in fold_results]
    full_test_accs = [result['test_acc_full_prototype'] for result in fold_results]
    full_test_losses = [result['test_loss_full_prototype'] for result in fold_results]
    
    print("\n" + "="*80)
    print("5-FOLD CROSS VALIDATION RESULTS (DUAL EVALUATION)")
    print("="*80)
    
    print("\nPer-fold results:")
    for result in fold_results:
        print(f"Fold {result['fold']}:")
        print(f"  Val Acc (episodic): {result['best_val_acc_episodic']:.4f}")
        print(f"  Test Acc (episodic): {result['test_acc_episodic']:.4f}")
        print(f"  Test Acc (full prototype): {result['test_acc_full_prototype']:.4f}")
        print(f"  Test samples (full): {result['num_test_samples_full']}")
    
    print(f"\nOverall Statistics:")
    print(f"Mean Validation Accuracy (episodic): {np.mean(val_accs):.4f} \u00b1 {np.std(val_accs):.4f}")
    print(f"Mean Test Accuracy (episodic): {np.mean(episodic_test_accs):.4f} \u00b1 {np.std(episodic_test_accs):.4f}")
    print(f"Mean Test Accuracy (full prototype): {np.mean(full_test_accs):.4f} \u00b1 {np.std(full_test_accs):.4f}")
    print(f"Mean Test Loss (episodic): {np.mean(episodic_test_losses):.4f} \u00b1 {np.std(episodic_test_losses):.4f}")
    print(f"Mean Test Loss (full prototype): {np.mean(full_test_losses):.4f} \u00b1 {np.std(full_test_losses):.4f}")
    
    print(f"\nBest Results:")
    print(f"Best Episodic Test Accuracy: {max(episodic_test_accs):.4f}")
    print(f"Best Full Prototype Test Accuracy: {max(full_test_accs):.4f}")
    print(f"Worst Episodic Test Accuracy: {min(episodic_test_accs):.4f}")
    print(f"Worst Full Prototype Test Accuracy: {min(full_test_accs):.4f}")
    
    # Save cross-validation results
    cv_results_df = pd.DataFrame(fold_results)
    cv_results_df.to_csv("5_fold_cv_results_dual_evaluation.csv", index=False)
    
    # Save summary statistics
    with open("5_fold_cv_summary_dual_evaluation.txt", "w") as f:
        f.write("5-Fold Cross Validation Summary (Dual Evaluation Methods)\n")
        f.write("="*70 + "\n\n")
        f.write("Training: Episodic learning (5-way few-shot)\n")
        f.write("Validation: Episodic evaluation\n")
        f.write("Testing: Both episodic and full prototype evaluation\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"Mean Validation Accuracy (episodic): {np.mean(val_accs):.4f} \u00b1 {np.std(val_accs):.4f}\n")
        f.write(f"Mean Test Accuracy (episodic): {np.mean(episodic_test_accs):.4f} \u00b1 {np.std(episodic_test_accs):.4f}\n")
        f.write(f"Mean Test Accuracy (full prototype): {np.mean(full_test_accs):.4f} \u00b1 {np.std(full_test_accs):.4f}\n")
        f.write(f"Mean Test Loss (episodic): {np.mean(episodic_test_losses):.4f} \u00b1 {np.std(episodic_test_losses):.4f}\n")
        f.write(f"Mean Test Loss (full prototype): {np.mean(full_test_losses):.4f} \u00b1 {np.std(full_test_losses):.4f}\n\n")
        
        f.write("Best Results:\n")
        f.write(f"Best Episodic Test Accuracy: {max(episodic_test_accs):.4f}\n")
        f.write(f"Best Full Prototype Test Accuracy: {max(full_test_accs):.4f}\n")
        f.write(f"Worst Episodic Test Accuracy: {min(episodic_test_accs):.4f}\n")
        f.write(f"Worst Full Prototype Test Accuracy: {min(full_test_accs):.4f}\n\n")
        
        f.write("Per-fold Results:\n")
        for result in fold_results:
            f.write(f"Fold {result['fold']}:\n")
            f.write(f"  Val Acc (episodic): {result['best_val_acc_episodic']:.4f}\n")
            f.write(f"  Test Acc (episodic): {result['test_acc_episodic']:.4f}\n")
            f.write(f"  Test Acc (full prototype): {result['test_acc_full_prototype']:.4f}\n")
            f.write(f"  Test samples (full): {result['num_test_samples_full']}\n\n")
    
    print(f"\nResults saved to: 5_fold_cv_results_dual_evaluation.csv")
    print(f"Summary saved to: 5_fold_cv_summary_dual_evaluation.txt")
    print(f"Individual classification reports saved as: classification_report_fold_X.txt")
    
    return fold_results

# --- 6. Main Execution Block ---
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Run 5-fold cross validation with dual evaluation methods
    results = run_5_fold_cross_validation_dual_evaluation()
