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

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

from pathlib import Path
from dataclasses import dataclass

import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


# ------------------ Config ------------------
SAMPLE_RATE = 16000
SEGMENT_LENGTH = 10 * SAMPLE_RATE
# Audio processing parameters
WINDOW_MS = 25
HOP_MS = 10
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_MS / 1000)  # 400 samples
HOP_SIZE = int(SAMPLE_RATE * HOP_MS / 1000)        # 160 samples

mel_kwargs = {
    "n_fft": 512,              # FFT size (power of 2, >= win_length)
    "win_length": WINDOW_SIZE, # 400 samples (25 ms)
    "hop_length": HOP_SIZE,    # 160 samples (10 ms)
    "n_mels": 64,
    "f_min": 0,
    "f_max": SAMPLE_RATE // 2  # Nyquist frequency
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

# ------------------ Training Params ------------------
@dataclass
class TrainingParams:
    num_epochs: int = 100
    batch_size: int = 4  # Very small for full spectrogram processing
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    patience: int = 15
    eval_every: int = 5
    train_csv_path: str = "train_log.csv"
    val_csv_path: str = "val_log.csv"
    best_model_path: str = "best_quantum_cnn_full_spec_model.pt"


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
        if max(onset, start) < min(offset, end):
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
    
    return 4  # Default to "no breathing"

# ------------------ Dataset Classes for Full Spectrograms ------------------
class BreathingSegmentDataset(Dataset):
    def __init__(self, audio_dir, label_dir, segment_length=SEGMENT_LENGTH, transform=None):
        self.data = []
        self.transform = MelSpectrogram(sample_rate=SAMPLE_RATE, **mel_kwargs)
        
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
                
                # Convert to mel-spectrogram and keep full resolution
                mel_spec = self.transform(segment)
                log_mel = torch.log(mel_spec + 1e-8).squeeze(0)
                
                # Resize to manageable size for quantum processing
                resized_mel = F.interpolate(
                    log_mel.unsqueeze(0).unsqueeze(0), 
                    size=(32, 32), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                
                self.data.append((resized_mel, label))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        mel_spec, label = self.data[idx]
        return mel_spec, label

class SMOTEDataset(Dataset):
    def __init__(self, dataset, random_state=42):
        self.original_dataset = dataset
        self.random_state = random_state
        self.data = []
        self.labels = []
        
        X_original = []
        y_original = []
        
        print("Extracting full mel-spectrograms from original dataset...")
        for i in range(len(self.original_dataset)):
            mel_spec, label = self.original_dataset[i]
            if i == 0:
                self.original_shape = mel_spec.shape
            
            X_original.append(mel_spec.flatten().numpy())
            y_original.append(label)
        
        X_original = np.array(X_original)
        y_original = np.array(y_original)
        
        print("Original class distribution:", Counter(y_original))
        
        print("Applying SMOTE resampling...")
        under_sampler = RandomUnderSampler(sampling_strategy={4: 500}, random_state=42)
        smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)

        pipeline = Pipeline([('under', under_sampler), ('over', smote)])
        X_resampled, y_resampled = pipeline.fit_resample(X_original, y_original)

        print("Resampled class distribution:", Counter(y_resampled))
        
        self._preprocess(X_resampled, y_resampled)
        
        print(f"SMOTE preprocessing complete. Dataset size: {len(self.data)} samples")

    def _preprocess(self, X, y):
        for i in range(len(y)):
            mel_spec = torch.tensor(X[i].reshape(self.original_shape), dtype=torch.float32)
            label = int(y[i])
            self.data.append(mel_spec)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ------------------ Quantum CNN Classifier ------------------
class SplitParallelQCNN(nn.Module):
    def __init__(self, n_qubits=16, n_splits=4, n_layers=2, num_classes=5):
        super(SplitParallelQCNN, self).__init__()
        self.n_qubits = n_qubits
        self.n_splits = n_splits
        self.qubits_per_split = n_qubits // n_splits
        self.n_layers = n_layers
        self.num_classes = num_classes
        
        # Create separate quantum devices for each split
        self.devices = [
            qml.device('default.qubit', wires=self.qubits_per_split) 
            for _ in range(n_splits)
        ]
        
        # Parameters for each parallel quantum circuit
        self.split_weights = nn.ParameterList([
            nn.Parameter(torch.randn(n_layers, self.qubits_per_split, 3, dtype=torch.float32) * 0.1)
            for _ in range(n_splits)
        ])
        
        # Classical layers for feature aggregation
        self.feature_dim = n_splits * self.qubits_per_split
        self.aggregation = nn.Sequential(
            nn.Linear(self.feature_dim, 128, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes, dtype=torch.float32)
        )
        
        # Create quantum circuits for each split
        self.quantum_circuits = self._create_parallel_circuits()
    
    def _create_parallel_circuits(self):
        """Create quantum circuits that can be used with threading"""
        circuits = []
        
        for i, device in enumerate(self.devices):
            @qml.qnode(device, interface='torch', diff_method='backprop')
            def circuit(weights, inputs, split_idx=i):
                # Data encoding with amplitude encoding
                qml.AmplitudeEmbedding(
                    features=inputs, 
                    wires=range(self.qubits_per_split),
                    normalize=True
                )
                
                # Variational layers
                for layer in range(weights.shape[0]):
                    # Rotation gates
                    for j in range(self.qubits_per_split):
                        qml.RX(weights[layer, j, 0], wires=j)
                        qml.RY(weights[layer, j, 1], wires=j)
                        qml.RZ(weights[layer, j, 2], wires=j)
                    
                    # Entangling gates
                    for j in range(self.qubits_per_split):
                        qml.CNOT(wires=[j, (j + 1) % self.qubits_per_split])
                
                # Measurements
                return [qml.expval(qml.PauliZ(j)) for j in range(self.qubits_per_split)]
            
            circuits.append(circuit)
        
        return circuits
    
    def forward(self, x, use_multiprocessing=False):
        """Forward pass with threading-based parallelization"""
        batch_size = x.shape[0]
        all_outputs = []
        
        # Process each sample in the batch
        for batch_idx in range(batch_size):
            sample = x[batch_idx].float()
            split_inputs = self._split_data_symmetrically(sample)
            
            # Use ThreadPoolExecutor for quantum circuits
            with ThreadPoolExecutor(max_workers=self.n_splits) as executor:
                futures = []
                
                for split_idx, (circuit, split_input, weights) in enumerate(
                    zip(self.quantum_circuits, split_inputs, self.split_weights)
                ):
                    future = executor.submit(circuit, weights, split_input)
                    futures.append(future)
                
                # Collect results
                parallel_results = []
                for future in futures:
                    result = future.result()
                    result_tensor = torch.stack(result).float()
                    parallel_results.append(result_tensor)
            
            # Combine parallel outputs
            combined_output = torch.cat(parallel_results)
            all_outputs.append(combined_output)
        
        quantum_features = torch.stack(all_outputs).float()
        output = self.aggregation(quantum_features)
        return output
    
    def _split_data_symmetrically(self, data):
        """Split input data based on translational symmetry"""
        # Flatten the mel-spectrogram data and ensure float32
        flat_data = data.flatten().float()
        
        # Ensure we have enough data points
        required_size = self.n_splits * (2 ** self.qubits_per_split)
        
        if len(flat_data) < required_size:
            # Pad with zeros
            padding = torch.zeros(required_size - len(flat_data), device=data.device, dtype=torch.float32)
            flat_data = torch.cat([flat_data, padding])
        else:
            # Truncate to required size
            flat_data = flat_data[:required_size]
        
        # Split data into chunks for each quantum circuit
        splits = []
        chunk_size = 2 ** self.qubits_per_split
        
        for i in range(self.n_splits):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            split_data = flat_data[start_idx:end_idx]
            
            # Normalize for amplitude encoding
            norm = torch.norm(split_data)
            if norm > 0:
                split_data = (split_data / norm).float()
            
            splits.append(split_data)
        
        return splits


class ParallelQCNNTrainer:
    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Set up parallel processing
        self._setup_parallel_environment()
    
    def _setup_parallel_environment(self):
        """Configure parallel processing environment"""
        # Set PyTorch threading
        torch.set_num_threads(torch.get_num_threads())
        torch.set_num_interop_threads(4)
        
        # Set environment variables for optimal performance
        import os
        os.environ['OMP_NUM_THREADS'] = str(torch.get_num_threads())
        os.environ['MKL_NUM_THREADS'] = str(torch.get_num_threads())
        
        print(f"Using {torch.get_num_threads()} CPU threads for training")
    
    def train_epoch(self, train_loader, use_multiprocessing=False):
        """Train for one epoch with parallel quantum processing"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with parallel quantum processing
            outputs = self.model(data, use_multiprocessing=use_multiprocessing)
            
            # Compute loss
            loss = self.criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Progress reporting
            if batch_idx % 10 == 0:
                batch_time = time.time() - start_time
                print(f'Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss={loss.item():.4f}, '
                      f'Acc={100.*correct/total:.2f}%, '
                      f'Time={batch_time:.2f}s')
                start_time = time.time()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, use_multiprocessing=False):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data, use_multiprocessing=use_multiprocessing)
                loss = self.criterion(outputs, target)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test(self, test_loader, use_multiprocessing=False, class_names=None):
        """Comprehensive test evaluation with detailed metrics"""
        print("\n" + "="*60)
        print("TESTING PHASE - Final Model Evaluation")
        print("="*60)
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.model(data, use_multiprocessing=use_multiprocessing)
                loss = self.criterion(outputs, target)
                
                # Get predictions and probabilities
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Store results
                total_loss += loss.item()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if batch_idx % 5 == 0:
                    print(f"Test batch {batch_idx}/{len(test_loader)} processed...")
        
        test_time = time.time() - start_time
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Print summary results
        print(f"\nTest Results Summary:")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        print(f"Test Time: {test_time:.2f} seconds")
        print(f"Total Test Samples: {len(all_targets)}")
        
        # Detailed classification report
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(set(all_targets)))]
        
        print(f"\nDetailed Classification Report:")
        print("-" * 50)
        report = classification_report(
            all_targets, all_predictions, 
            target_names=class_names, 
            digits=4
        )
        print(report)
        
        # Confusion Matrix
        self._plot_confusion_matrix(all_targets, all_predictions, class_names)
        
        # Per-class accuracy
        self._print_per_class_accuracy(all_targets, all_predictions, class_names)
        
        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'classification_report': report
        }
    
    def _plot_confusion_matrix(self, targets, predictions, class_names):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nConfusion Matrix saved as 'test_confusion_matrix.png'")
    
    def _print_per_class_accuracy(self, targets, predictions, class_names):
        """Print per-class accuracy breakdown"""
        cm = confusion_matrix(targets, predictions)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        print(f"\nPer-Class Accuracy:")
        print("-" * 40)
        for i, (class_name, acc) in enumerate(zip(class_names, per_class_acc)):
            print(f"{class_name:25}: {acc*100:.2f}%")
    
    def train(self, train_loader, val_loader, epochs=100, use_multiprocessing=False):
        """Complete training loop with parallel processing"""
        print(f"Starting parallel QCNN training with {self.model.n_splits} splits")
        print(f"Multiprocessing: {use_multiprocessing}")
        print(f"Qubits per split: {self.model.qubits_per_split}")
        start_time = time.time()
        
        best_val_acc = 0.0
        training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, use_multiprocessing)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, use_multiprocessing)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'training_history': training_history
                }, 'best_parallel_qcnn.pth')
                print(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
            
            # Record history
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Best Val Acc: {best_val_acc:.2f}%")
        
        elapsed_time = time.time() - start_time
        print(f"\nQuantum training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        training_history['elapsed_time'] = elapsed_time
        save_history_to_csv(training_history, 'training_history_final.csv')
        return training_history
    
    def load_best_model(self, model_path='best_parallel_qcnn.pth'):
        """Load the best saved model for testing"""
        print(f"Loading best model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully! Best validation accuracy was: {checkpoint['best_val_acc']:.2f}%")
        return checkpoint

def plot_training_history(history):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Training Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history['train_acc'], label='Training Accuracy', color='blue')
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_history_to_csv(history, filename='training_history.csv'):
    """Save training history to CSV file"""
    df = pd.DataFrame(history)
    df['epoch'] = range(1, len(df) + 1)
    # Reorder columns to have epoch first
    cols = ['epoch'] + [col for col in df.columns if col != 'epoch']
    df = df[cols]
    df.to_csv(filename, index=False)
    print(f"Training history saved to {filename}")
    return filename

def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Dataset setup
    dataset = BreathingSegmentDataset("audio", "strong_labels")
    
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
    breathing_class_names = []
    for class_idx, count in class_counts.items():
        class_name = [k for k, v in BREATHING_LABELS.items() if v == class_idx][0]
        breathing_class_names.append(class_name)
        print(f"  {class_name}: {count} samples")

    # Apply SMOTE to training set
    train_set_resampled = SMOTEDataset(train_set)

    # Model parameters
    n_qubits = 16
    n_splits = 4
    n_layers = 2
    num_classes = 5
    
    # Create model
    model = SplitParallelQCNN(
        n_qubits=n_qubits,
        n_splits=n_splits,
        n_layers=n_layers,
        num_classes=num_classes
    )
    
    # Create trainer
    trainer = ParallelQCNNTrainer(model, device='cpu', learning_rate=0.001)
    
    # Optimize DataLoader for parallel processing
    num_cores = torch.get_num_threads()
    num_workers = min(num_cores - 2, 16)
    batch_size = 32
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_set_resampled,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Train the model
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        use_multiprocessing=False
    )
    
    print("Training completed!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model and test
    checkpoint = trainer.load_best_model()
    
    # Comprehensive test evaluation
    test_results = trainer.test(
        test_loader=test_loader,
        use_multiprocessing=False,
        class_names=list(BREATHING_LABELS.keys())
    )
    
    # Save test results
    test_summary = {
        'test_accuracy': test_results['test_accuracy'],
        'test_loss': test_results['test_loss'],
        'best_val_accuracy': checkpoint['best_val_acc'],
        'model_parameters': {
            'n_qubits': n_qubits,
            'n_splits': n_splits,
            'n_layers': n_layers,
            'num_classes': num_classes
        }
    }
    
    # Save results to file
    import json
    with open('test_results_summary.json', 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {checkpoint['best_val_acc']:.2f}%")
    print(f"Final Test Accuracy: {test_results['test_accuracy']*100:.2f}%")
    print(f"Test results saved to 'test_results_summary.json'")

if __name__ == "__main__":
    main()
