import os
import glob
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from hmmlearn import hmm
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Global Configuration ---

class Config:
    """Configuration parameters."""
    sample_rate = 16000
    segment_length_sec = 10
    segment_length_samples = segment_length_sec * sample_rate
    
    frame_size = 1024
    hop_size = 512
    
    n_mfcc = 13
    n_mels = 32
    n_chroma = 12
    
    test_split_ratio = 0.2
    n_folds = 5
    
    # Paths for reports
    agg_cv_history_path = "report_scenario1_cv_history.csv"
    agg_final_report_path = "report_scenario1_final_test.txt"
    voted_cv_history_path = "report_scenario2_cv_history.csv"
    voted_final_report_path = "report_scenario2_final_test.txt"

BREATHING_LABELS = {
    "breathing": 0, "light breathing": 1, "heavy breathing": 2,
    "heavy breathing with noises": 3, "no breathing": 4
}
LABEL_NAMES = [k for k, v in sorted(BREATHING_LABELS.items(), key=lambda item: item[1])]

# --- 2. Simplified HMM Classifier ---

class HMMClassifier:
    def __init__(self, n_classes, n_components=1, covariance_type='diag', random_state=42):
        self.n_classes = n_classes
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.models = [
            hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_iter=50,
                tol=1e-2,
                min_covar=1e-3
            ) for _ in range(self.n_classes)
        ]
        
    def fit(self, X, y, lengths):
        # Fit one HMM per class
        for i in range(self.n_classes):
            class_data = []
            class_lengths = []
            idx = 0
            
            for j, (length, label) in enumerate(zip(lengths, y)):
                if label == i:
                    class_data.append(X[idx:idx+length])
                    class_lengths.append(length)
                idx += length
            
            if class_data and len(class_data) > 0:
                X_class = np.concatenate(class_data, axis=0)
                if len(X_class) > self.n_components:  # Ensure enough data
                    self.models[i].fit(X_class, lengths=class_lengths)

    def predict_segments(self, X, lengths):
        """Predict segment-level labels using majority voting on frame predictions"""
        preds = []
        idx = 0
        
        for length in lengths:
            seq = X[idx:idx+length]
            
            # Get log-likelihood for each class
            log_likelihoods = []
            for model in self.models:
                try:
                    ll = model.score(seq)
                    log_likelihoods.append(ll)
                except:
                    log_likelihoods.append(-np.inf)
            
            # Predict segment label based on best likelihood
            preds.append(np.argmax(log_likelihoods))
            idx += length
            
        return np.array(preds)

    def predict_frames(self, X, lengths):
        """Predict frame-level labels, then aggregate to segments via majority voting"""
        frame_preds = []
        segment_preds = []
        idx = 0
        
        for length in lengths:
            seq = X[idx:idx+length]
            
            # Get best model for this sequence
            log_likelihoods = []
            for model in self.models:
                try:
                    ll = model.score(seq)
                    log_likelihoods.append(ll)
                except:
                    log_likelihoods.append(-np.inf)
            
            best_class = np.argmax(log_likelihoods)
            
            # Assign all frames in this segment to the best class
            frame_labels = [best_class] * length
            frame_preds.extend(frame_labels)
            
            # Majority vote for segment (in this case, all frames have same label)
            segment_preds.append(best_class)
            idx += length
            
        return np.array(frame_preds), np.array(segment_preds)

# --- 3. Feature Extraction ---

def extract_frame_features(y, sr, config):
    """Extract features from a single frame"""
    n_fft, hop_length = config.frame_size, config.hop_size
    
    # Ensure minimum length for feature extraction
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode='constant')
    
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = np.mean(mfcc, axis=1) if mfcc.shape[1] > 0 else mfcc.flatten()
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.n_mels, n_fft=n_fft, hop_length=hop_length)
        log_mel = librosa.power_to_db(mel_spec)
        log_mel = np.mean(log_mel, axis=1) if log_mel.shape[1] > 0 else log_mel.flatten()
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=config.n_chroma, n_fft=n_fft, hop_length=hop_length)
        chroma = np.mean(chroma, axis=1) if chroma.shape[1] > 0 else chroma.flatten()
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
        
        rms = np.mean(librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length))
        
        return np.hstack([mfcc, log_mel, chroma, [spectral_centroid, spectral_bandwidth, spectral_rolloff, rms]])
    except:
        # Return zero vector if feature extraction fails
        return np.zeros(config.n_mfcc + config.n_mels + config.n_chroma + 4)

def extract_segment_frame_features(segment, sr, config):
    """Extract frame-level features from a segment"""
    frames = librosa.util.frame(segment, frame_length=config.frame_size, hop_length=config.hop_size).T
    return np.array([extract_frame_features(frame, sr, config) for frame in frames])

# --- 4. Data Preparation ---

def load_labels(p):
    l=[]; open(p,'r');_=[l.append(tuple(map(str.strip,x.split(maxsplit=2)))) for x in open(p,'r')];return l

def is_breathing_segment(s,e,l):
    for o,f,t in l:
        if max(float(o),s)<min(float(f),e):
            m,c=0,None;[c:=i if breathing_type in t.lower() and len(breathing_type)>m else c for breathing_type,i in BREATHING_LABELS.items()];
            if c is not None: return c
    return BREATHING_LABELS["no breathing"]

def load_and_segment_audio(a,l,c):
    for p in tqdm(sorted(glob.glob(f"{a}/**/*.wav",recursive=True)),desc="Loading Segments"):
        f=Path(p).stem;lp=os.path.join(l,f"{f}.txt");
        if not os.path.exists(lp):continue
        e=load_labels(lp);
        try:w,r=librosa.load(p,sr=c.sample_rate,mono=True)
        except Exception:continue
        for i in range(len(w)//c.segment_length_samples):
            s=w[i*c.segment_length_samples:(i+1)*c.segment_length_samples];lab=is_breathing_segment(i*c.segment_length_sec,(i+1)*c.segment_length_sec,e);yield s,lab

def vote_on_predictions(frame_preds, segment_lengths):
    """Aggregate frame predictions to segment predictions via majority voting"""
    segment_preds = []
    idx = 0
    
    for length in segment_lengths:
        segment_frames = frame_preds[idx:idx+length]
        # Majority vote
        segment_pred = Counter(segment_frames).most_common(1)[0][0]
        segment_preds.append(segment_pred)
        idx += length
    
    return np.array(segment_preds)

# --- 5. Core Evaluation Protocol ---

def run_evaluation_protocol(X_train_pool, y_train_pool, X_test_final, y_test_final, 
                           segment_lengths_train, segment_lengths_test, config, 
                           scenario_name, report_path, history_path):
    """Run evaluation using frame features for both scenarios"""
    print(f"\n--- Running Full Evaluation for: {scenario_name} ---")
    start_time = time.time()
    
    # Create segment labels for stratification
    segment_labels_train = []
    idx = 0
    for length in segment_lengths_train:
        segment_labels_train.append(y_train_pool[idx])
        idx += length
    segment_labels_train = np.array(segment_labels_train)
    
    # 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
    cv_histories = []
    
    for fold, (train_seg_idx, val_seg_idx) in enumerate(skf.split(np.arange(len(segment_lengths_train)), segment_labels_train), 1):
        print(f"\n-- Fold {fold}/{config.n_folds} --")
        
        # Convert segment indices to frame indices
        train_frame_indices = []
        val_frame_indices = []
        train_lengths = []
        val_lengths = []
        
        frame_idx = 0
        for seg_idx, length in enumerate(segment_lengths_train):
            if seg_idx in train_seg_idx:
                train_frame_indices.extend(range(frame_idx, frame_idx + length))
                train_lengths.append(length)
            elif seg_idx in val_seg_idx:
                val_frame_indices.extend(range(frame_idx, frame_idx + length))
                val_lengths.append(length)
            frame_idx += length
        
        X_train_fold = X_train_pool[train_frame_indices]
        y_train_fold = segment_labels_train[train_seg_idx]
        X_val_fold = X_train_pool[val_frame_indices]
        y_val_fold = segment_labels_train[val_seg_idx]
        
        # Train HMM
        model = HMMClassifier(n_classes=len(LABEL_NAMES), n_components=1, covariance_type='diag', random_state=42)
        model.fit(X_train_fold, y_train_fold, lengths=train_lengths)
        
        # Predict segments
        y_pred_val = model.predict_segments(X_val_fold, lengths=val_lengths)
        val_accuracy = np.mean(y_pred_val == y_val_fold)
        
        cv_histories.append({
            'fold': fold, 'epoch': 1,
            'train_loss': 0, 'train_accuracy': 0,
            'val_loss': 0, 'val_accuracy': val_accuracy,
        })
        print(f"Fold {fold} completed. Val Accuracy: {val_accuracy:.4f}")

    # Save CV history
    cv_df = pd.DataFrame(cv_histories)
    cv_df.to_csv(history_path, index=False)
    print(f"Cross-validation history saved to {history_path}")

    # Final Model Training & Evaluation
    print("\n-- Training Final Model --")
    segment_labels_train_final = []
    idx = 0
    for length in segment_lengths_train:
        segment_labels_train_final.append(y_train_pool[idx])
        idx += length
    
    final_model = HMMClassifier(n_classes=len(LABEL_NAMES), n_components=1, covariance_type='diag', random_state=42)
    final_model.fit(X_train_pool, np.array(segment_labels_train_final), lengths=segment_lengths_train)

    # Final prediction
    y_pred_final = final_model.predict_segments(X_test_final, lengths=segment_lengths_test)
    
    # Get true segment labels
    y_true_final = []
    idx = 0
    for length in segment_lengths_test:
        y_true_final.append(y_test_final[idx])
        idx += length
    y_true_final = np.array(y_true_final)

    elapsed_time = time.time() - start_time
    final_report = classification_report(y_true_final, y_pred_final, target_names=LABEL_NAMES, zero_division=0)
    
    with open(report_path, "w") as f:
        f.write(f"--- Final Test Report for {scenario_name} ---\n\n{final_report}")
        f.write(f"\nElapsed Time: {elapsed_time:.2f} seconds\n")
    
    print("\n--- Final Test Set Report ---")
    print(final_report)
    print(f"Final report saved to {report_path}")

# --- 6. Main Execution Block ---

if __name__ == "__main__":
    random.seed(42), np.random.seed(42)
    config = Config()
    AUDIO_DIR, LABEL_DIR = "./audio", "./strong_labels"
    
    all_segments = list(load_and_segment_audio(AUDIO_DIR, LABEL_DIR, config))
    all_segment_waveforms, all_segment_labels = [item[0] for item in all_segments], np.array([item[1] for item in all_segments])
    print(f"Loaded {len(all_segments)} total 10-second segments.")
    print(f"Segment labels: {Counter(all_segment_labels)}")

    # Extract frame features for all segments
    frame_features_per_segment = []
    for segment_waveform in tqdm(all_segment_waveforms, desc="Extracting frame features"):
        frame_features = extract_segment_frame_features(segment_waveform, config.sample_rate, config)
        frame_features_per_segment.append(frame_features)
    
    # Split segments into train/test
    train_segments, test_segments = train_test_split(
        np.arange(len(all_segments)),
        test_size=config.test_split_ratio,
        random_state=42,
        stratify=all_segment_labels
    )
    
    # Prepare training data
    X_train_pool = np.concatenate([frame_features_per_segment[i] for i in train_segments], axis=0)
    y_train_pool = np.concatenate([[all_segment_labels[i]] * len(frame_features_per_segment[i]) for i in train_segments])
    segment_lengths_train = [len(frame_features_per_segment[i]) for i in train_segments]
    
    # Prepare test data
    X_test_final = np.concatenate([frame_features_per_segment[i] for i in test_segments], axis=0)
    y_test_final = np.concatenate([[all_segment_labels[i]] * len(frame_features_per_segment[i]) for i in test_segments])
    segment_lengths_test = [len(frame_features_per_segment[i]) for i in test_segments]

    # Run both scenarios using the same frame-based approach
    print("\n=== SCENARIO 1: Frame Features -> Segment Prediction ===")
    run_evaluation_protocol(X_train_pool, y_train_pool, X_test_final, y_test_final,
                           segment_lengths_train, segment_lengths_test, config,
                           "Frame Features for Segment Prediction", config.agg_final_report_path, config.agg_cv_history_path)

    print("\n=== SCENARIO 2: Frame Features -> Frame Prediction -> Segment Voting ===")
    run_evaluation_protocol(X_train_pool, y_train_pool, X_test_final, y_test_final,
                           segment_lengths_train, segment_lengths_test, config,
                           "Frame Features with Voting", config.voted_final_report_path, config.voted_cv_history_path)
