import os
import time
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.transforms import MelSpectrogram
from torchvggish import vggish, vggish_input

from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

from pathlib import Path
from dataclasses import dataclass

from transformers import WhisperProcessor, WhisperModel, ASTFeatureExtractor, ASTModel


# ------------------ Config ------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ------------------ New Training Params ------------------
# Training Parameters
class TrainingParams:
    num_epochs = 1000       # Total training episodes
    eval_every = 10          # Evaluate on validation set every N episodes
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

    samples_per_class = 3  # Number of samples per class in each batch
    num_classes = 5        # Your breathing classes
    balanced_batch_size = samples_per_class * num_classes  # 15 samples per batch

    best_model_path = "best_hybrid_model.pth"
    train_log_path = "training_log.csv"
    eval_log_path = "validation_log.csv"
    final_report_path = "final_evaluation_report.txt"
    final_cm_path = "final_confusion_matrix.csv"

# ------------------ Utils ------------------
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

# ------------------ Encoder ------------------
class VGGishEncoder(nn.Module):
    def __init__(self, num_classes=5, params=None):
        super().__init__()
        self.params = params if params else TrainingParams()
        self.vggish = vggish(postprocess=False)
        self.vggish.eval()
        for param in self.vggish.parameters():
            param.requires_grad = False
        self.proj = nn.Linear(128, self.params.embedding_dim)  # VGGish outputs 128-dim features
        self.classifier = PrototypicalNetworkClassifier(
            embedding_dim=self.params.embedding_dim,
            num_classes=num_classes,
            distance_metric='euclidean'  # or 'cosine', 'squared_euclidean'
        )

    def forward(self, x, labels=None, mode='train'):
        # x shape: [B, 1, T], where T is audio length
        embeddings = []
        for i in range(x.size(0)):
            waveform = x[i].squeeze().cpu().numpy()  # [T]
            examples = vggish_input.waveform_to_examples(waveform, self.params.sample_rate)  # [N, 96, 64]

            examples = examples.float().to(x.device)  # [N, 1, 96, 64]
            with torch.no_grad():
                feat = self.vggish(examples)  # [N, 128]
            embeddings.append(feat.mean(dim=0))  # average over time windows

        embeddings = torch.stack(embeddings).to(x.device)  # [B, 128]
        embeddings = self.proj(embeddings)  # [B, out_dim]
        return self.classifier(embeddings, labels, mode)  # [B, num_classes]

class WhisperEncoder(nn.Module):
    def __init__(self, model_name="openai/whisper-small", num_classes=5, params=None):
        super().__init__()
        self.params = params if params else TrainingParams()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name)
        self.model.eval()  # Freeze the encoder
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Projection layer to match desired output dim
        self.proj = nn.Linear(self.model.config.d_model, self.params.embedding_dim) 
        self.classifier = PrototypicalNetworkClassifier(
            embedding_dim=self.params.embedding_dim,
            num_classes=num_classes,
            distance_metric='euclidean'
        ) 

    def forward(self, x, labels=None, mode='train'):
        # x: [B, 1, T] in waveform format
        embeddings = []
        for i in range(x.size(0)):
            waveform = x[i].squeeze().cpu()
            input_features = self.processor.feature_extractor(
                waveform.numpy(), sampling_rate=self.params.sample_rate, return_tensors="pt"
            ).input_features  # shape: [1, 80, 3000]

            with torch.no_grad():
                encoder_outputs = self.model.encoder(input_features.to(x.device))
                # Mean-pooling across time dimension
                emb = encoder_outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]
                embeddings.append(emb.squeeze(0))

        embeddings = torch.stack(embeddings)  # [B, hidden_dim]
        embeddings = self.proj(embeddings)  # [B, out_dim]
        return self.classifier(embeddings, labels, mode)  # [B, num_classes]

class ASTEncoder(nn.Module):
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", num_classes=5, params=None):
        super().__init__()
        self.params = params if params else TrainingParams()
        
        # Load pre-trained AST model and feature extractor from Hugging Face
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name)
        
        # Freeze the parameters of the pre-trained model to use it as a feature extractor
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Projection layer to map the AST output to your desired embedding dimension
        # The hidden size of the base AST model is 768
        self.proj = nn.Linear(self.model.config.hidden_size, self.params.embedding_dim)
        
        # Use the same Prototypical Network for the final classification
        self.classifier = PrototypicalNetworkClassifier(
            embedding_dim=self.params.embedding_dim,
            num_classes=num_classes,
            distance_metric='euclidean'
        )

    def forward(self, x, labels=None, mode='train'):
        # x is a batch of raw waveforms with shape [B, 1, T]
        embeddings = []
        
        # Process each waveform in the batch individually
        for i in range(x.size(0)):
            waveform = x[i].squeeze().cpu().numpy()
            
            # Use the feature extractor to convert the waveform to a spectrogram
            inputs = self.feature_extractor(
                waveform, 
                sampling_rate=self.params.sample_rate, 
                return_tensors="pt"
            )
            
            # Get embeddings from the AST model
            with torch.no_grad():
                # Ensure input tensors are on the correct device
                input_values = inputs.input_values.to(x.device)
                outputs = self.model(input_values)

            # The [CLS] token embedding is used as the representation for the entire audio clip
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.squeeze(0))
            
        # Stack the list of embeddings into a single tensor
        embeddings = torch.stack(embeddings)  # Shape: [B, hidden_dim]
        
        # Project the embeddings to the target dimension
        projected_embeddings = self.proj(embeddings)  # Shape: [B, embedding_dim]
        
        # Pass the final embeddings to the prototypical classifier
        return self.classifier(projected_embeddings, labels, mode)


class PrototypicalNetworkClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, distance_metric='euclidean'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.distance_metric = distance_metric
        
        # Store prototypes for each class (will be computed during training)
        self.register_buffer('prototypes', torch.zeros(num_classes, embedding_dim))
        self.prototype_counts = torch.zeros(num_classes)
        
        # Optional: learnable temperature parameter for softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
    def compute_prototypes(self, embeddings, labels):
        """Compute class prototypes as mean of embeddings for each class"""
        prototypes = torch.zeros(self.num_classes, self.embedding_dim, device=embeddings.device)
        
        for class_idx in range(self.num_classes):
            class_mask = (labels == class_idx)
            if class_mask.sum() > 0:
                # Compute mean embedding for this class
                prototypes[class_idx] = embeddings[class_mask].mean(dim=0)
            else:
                # If no examples for this class, use zero vector or previous prototype
                print(f"Warning: No examples for class {class_idx}, using zero vector.")
                prototypes[class_idx] = self.prototypes[class_idx].clone()
        
        return prototypes
    
    def compute_distances(self, query_embeddings, prototypes):
        """Compute distances between query embeddings and prototypes"""
        if self.distance_metric == 'euclidean':
            # Euclidean distance: ||query - prototype||_2
            distances = torch.cdist(query_embeddings, prototypes, p=2)
        elif self.distance_metric == 'cosine':
            # Cosine similarity converted to distance
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            similarities = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarities
        elif self.distance_metric == 'squared_euclidean':
            # Squared Euclidean distance
            distances = torch.cdist(query_embeddings, prototypes, p=2) ** 2
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def forward(self, embeddings, labels=None, mode='train'):
        """
        Forward pass for prototypical network classifier
        
        Args:
            embeddings: [batch_size, embedding_dim] - encoded features
            labels: [batch_size] - ground truth labels (required for training)
            mode: 'train' or 'eval' - determines behavior
        
        Returns:
            logits: [batch_size, num_classes] - classification logits
        """
        batch_size = embeddings.size(0)
        
        if mode == 'train' and labels is not None:
            # Training mode: compute prototypes from current batch
            prototypes = self.compute_prototypes(embeddings, labels)
            
            # Update running prototypes with exponential moving average
            with torch.no_grad():
                alpha = 0.1  # Update rate
                self.prototypes = (1 - alpha) * self.prototypes + alpha * prototypes
            
        else:
            # Evaluation mode: use stored prototypes
            prototypes = self.prototypes
        
        # Compute distances from each embedding to all prototypes
        distances = self.compute_distances(embeddings, prototypes)
        
        # Convert distances to logits (negative distances with temperature scaling)
        logits = -distances * self.temperature
        
        return logits
    
    def update_prototypes(self, embeddings, labels):
        """Explicitly update prototypes (useful for episodic training)"""
        with torch.no_grad():
            new_prototypes = self.compute_prototypes(embeddings, labels)
            self.prototypes.copy_(new_prototypes)


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

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params.num_epochs} [Train]")
        
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs, labels, mode='train')
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Calculate running averages
            current_loss = total_loss / total
            current_acc = 100 * correct / total
            
            # Update progress bar with current metrics
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Batch': f'{batch_idx+1}/{len(train_loader)}'
            })
        
        train_loss = total_loss / total
        train_acc = 100 * correct / total
        
        train_metrics.append({'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc})

        # Only evaluate every n epochs
        if (epoch + 1) % params.eval_every == 0 or epoch == params.num_epochs - 1:
            model.eval()
            val_total_loss = 0.0
            val_correct = 0
            val_total = 0

            # Create progress bar for validation batches
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{params.num_epochs} [Val]")
            
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_pbar):
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    outputs = model(inputs, labels=None, mode='eval')
                    loss = criterion(outputs, labels)
                    
                    val_total_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Calculate running validation metrics
                    current_val_loss = val_total_loss / val_total
                    current_val_acc = 100 * val_correct / val_total
                    
                    # Update validation progress bar
                    val_pbar.set_postfix({
                        'Loss': f'{current_val_loss:.4f}',
                        'Acc': f'{current_val_acc:.2f}%',
                        'Batch': f'{batch_idx+1}/{len(val_loader)}'
                    })
            
            val_acc = 100 * val_correct / val_total
            val_loss = val_total_loss / val_total
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
                torch.save(model.state_dict(), params.best_model_path)
            else:
                no_improve += 1

            if no_improve >= params.patience:
                print("Early stopping")
                break

    # Save metrics to CSV using pandas
    pd.DataFrame(train_metrics).to_csv(params.train_log_path, index=False)
    pd.DataFrame(val_metrics).to_csv(params.eval_log_path, index=False)
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Best model saved to {params.best_model_path} with val acc {best_acc:.4f}, val loss {best_loss:.4f}")
    with open("train_log.txt", "w") as f:
        f.write(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
        f.write(f"Best validation accuracy: {best_acc:.2f}% at epoch {epoch+1}\n")
        f.write(f"Best validation loss: {best_loss:.4f}\n")

class BalancedSampler(Sampler):
    """
    Advanced balanced sampler with multiple strategies for handling class imbalance.
    """
    
    def __init__(self, dataset, samples_per_class=4, strategy='oversample', 
                 num_batches=1000, min_samples_threshold=10):
        self.dataset = dataset
        self.samples_per_class = samples_per_class
        self.strategy = strategy
        self.num_batches = num_batches
        self.min_samples_threshold = min_samples_threshold
        
        # Group indices by class
        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.class_to_indices[label].append(idx)
        
        # Print class distribution
        print("Class distribution in sampler:")
        for class_label, indices in self.class_to_indices.items():
            class_name = [k for k, v in BREATHING_LABELS.items() if v == class_label][0]
            print(f"  {class_name}: {len(indices)} samples")
    
    def __iter__(self):
        for batch_idx in range(self.num_batches):
            batch_indices = []
            
            for class_label in range(len(BREATHING_LABELS)):
                if class_label in self.class_to_indices:
                    class_indices = self.class_to_indices[class_label]
                    
                    if self.strategy == 'oversample':
                        # Oversample minority classes
                        sampled_indices = np.random.choice(
                            class_indices, 
                            size=self.samples_per_class, 
                            replace=len(class_indices) < self.samples_per_class
                        )
                    elif self.strategy == 'undersample':
                        # Undersample majority classes
                        if len(class_indices) >= self.samples_per_class:
                            sampled_indices = np.random.choice(
                                class_indices, 
                                size=self.samples_per_class, 
                                replace=False
                            )
                        else:
                            sampled_indices = np.random.choice(
                                class_indices, 
                                size=self.samples_per_class, 
                                replace=True
                            )
                    else:  # 'adaptive'
                        # Adaptive sampling based on class size
                        if len(class_indices) < self.min_samples_threshold:
                            # Oversample very small classes
                            sampled_indices = np.random.choice(
                                class_indices, 
                                size=self.samples_per_class, 
                                replace=True
                            )
                        else:
                            # Regular sampling for larger classes
                            sampled_indices = np.random.choice(
                                class_indices, 
                                size=self.samples_per_class, 
                                replace=False
                            )
                    
                    batch_indices.extend(sampled_indices)
                else:
                    # Handle missing classes by sampling from available data
                    all_indices = []
                    for indices in self.class_to_indices.values():
                        all_indices.extend(indices)
                    sampled_indices = np.random.choice(
                        all_indices, 
                        size=self.samples_per_class, 
                        replace=True
                    )
                    batch_indices.extend(sampled_indices)
            
            # Shuffle to avoid class ordering within batch
            random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        return self.num_batches


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
    params = TrainingParams()
    dataset = BreathingSegmentDataset("audio", "strong_labels", params=params, 
                                      augment_minority_classes=True, target_min_samples=100)
    
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

    # Create DataLoaders
    train_sampler = BalancedSampler(
        dataset=train_set,
        samples_per_class=params.samples_per_class,
        strategy='oversample',  # Best for your imbalanced data
        num_batches=params.batch_size,
        min_samples_threshold=50
    )

    # Create DataLoader with batch sampler
    train_loader = DataLoader(
        train_set, 
        batch_sampler=train_sampler,  # Use batch_sampler instead of sampler
        num_workers=0  # Set to 0 for debugging, increase for performance
    )
    # train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params.batch_size)
    test_loader = DataLoader(test_set, batch_size=params.batch_size)
    
    # Initialize model
    # model = VGGishEncoder(num_classes=5, params=params).to(DEVICE)

    # print("Initializing Whisper encoder...")
    # model = WhisperEncoder(
    #     model_name="openai/whisper-base",  # You can also try "openai/whisper-tiny" for faster training
    #     num_classes=5, 
    #     params=params
    # ).to(DEVICE)

    print("Initializing AST encoder...")
    model = ASTEncoder(
        model_name="MIT/ast-finetuned-audioset-10-10-0.4593",  # Pre-trained AST model
        num_classes=5, 
        params=params
    ).to(DEVICE)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Train
    train_model(train_loader, val_loader, model, params)
    
    # Final evaluation
    model.load_state_dict(torch.load(params.best_model_path, weights_only=True))
    evaluate_model(test_loader, model, nn.CrossEntropyLoss())
