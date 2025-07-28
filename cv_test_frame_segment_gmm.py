import os
import glob
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from sklearn.mixture import GaussianMixture
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

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
    
    xgb_params = {
        'objective': 'multi:softprob', 'use_label_encoder': False,
        'n_estimators': 1000,  # Increased for more detailed learning curves
        'learning_rate': 0.1, 'max_depth': 5,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
        'eval_metric': ['mlogloss', 'merror'],
    }
    
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
# --- 2. Feature Extraction (Unchanged from previous correct version) ---
# Functions: extract_true_frame_features, extract_aggregated_segment_features, extract_frame_level_features

class GMMClassifier:
    def __init__(self, n_classes, n_components=2, covariance_type='full', random_state=42):
        self.n_classes = n_classes
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.models = [GaussianMixture(
            n_components=self.n_components, 
            covariance_type=self.covariance_type,
            random_state=self.random_state) for _ in range(self.n_classes)]
        self.thresholds = np.zeros(self.n_classes)
        
    def fit(self, X, y):
        # Fit a GMM to each class
        for i in range(self.n_classes):
            X_class = X[y == i]
            self.models[i].fit(X_class)
        # Compute thresholds using training data (can be improved using validation set)
        scores = [self.models[i].score_samples(X[y == i]) for i in range(self.n_classes)]
        self.thresholds = np.array([np.percentile(s, 5) for s in scores])  # 5th percentile as threshold

    def predict(self, X):
        # Compute log-likelihood for each class
        log_likelihoods = np.array([model.score_samples(X) for model in self.models]).T  # shape: (n_samples, n_classes)
        # Compare to thresholds
        above_thresh = (log_likelihoods >= self.thresholds)
        # Assign to class with highest likelihood above threshold, else to class with highest likelihood
        preds = np.argmax(log_likelihoods, axis=1)
        for i, row in enumerate(above_thresh):
            if not np.any(row):
                preds[i] = np.argmax(log_likelihoods[i])
            else:
                preds[i] = np.where(row)[0][np.argmax(log_likelihoods[i][row])]
        return preds

def extract_true_frame_features(y, sr, config):
    n_fft, hop_length = config.frame_size, config.hop_size
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=config.n_chroma, n_fft=n_fft, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr, chroma=chroma)[:, 0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.n_mfcc, n_fft=n_fft, hop_length=hop_length)[:, 0]
    log_mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.n_mels, n_fft=n_fft, hop_length=hop_length))[:, 0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[:, 0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).flatten()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).flatten()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).flatten()
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length).flatten()
    peak = np.array([np.max(np.abs(y))])
    return np.hstack([mfcc, log_mel_spec, chroma[:, 0], spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff, tonnetz, rms, librosa.amplitude_to_db(rms, ref=np.max).flatten(), peak])

def extract_aggregated_segment_features(segment, sr, config):
    mfcc, log_mel_spec, chroma, rms = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=config.n_mfcc), librosa.power_to_db(librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=config.n_mels)), librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=config.n_chroma), librosa.feature.rms(y=segment)
    return np.hstack([np.mean(mfcc, axis=1), np.mean(log_mel_spec, axis=1), np.mean(chroma, axis=1), np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr), axis=1), np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr), axis=1), np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr), axis=1), np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr), axis=1), np.mean(librosa.feature.tonnetz(y=segment, sr=sr), axis=1), np.mean(rms, axis=1), np.mean(librosa.amplitude_to_db(rms, ref=np.max), axis=1), np.max(np.abs(segment))])

def extract_frame_level_features(segment, sr, config):
    frames = librosa.util.frame(segment, frame_length=config.frame_size, hop_length=config.hop_size).T
    return np.array([extract_true_frame_features(frame, sr, config) for frame in frames])

# --- 3. Data Preparation (Unchanged) ---
# Functions: load_labels, is_breathing_segment, load_and_segment_audio, vote_on_predictions
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

def vote_on_predictions(p,t,c):
    f=len(librosa.util.frame(np.zeros(c.segment_length_samples),frame_length=c.frame_size,hop_length=c.hop_size).T);n=len(t)//f;ps,ts=[],[];
    for i in range(n):s,e=i*f,(i+1)*f;ps.append(Counter(p[s:e]).most_common(1)[0][0]);ts.append(t[s])
    return np.array(ps),np.array(ts)

# --- 4. NEW: Core Evaluation Protocol ---

def run_evaluation_protocol(X_train_pool, y_train_pool, X_test_final, y_test_final, config, scenario_name, report_path, history_path, is_frame_scenario=False):
    """
    Runs the full 5-fold CV + final test evaluation and saves all results.
    """
    print(f"\n--- Running Full Evaluation for: {scenario_name} ---")
    start_time = time.time()
    
    # --- 5-Fold Cross-Validation ---
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
    cv_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_pool, y_train_pool), 1):
        print(f"\n-- Fold {fold}/{config.n_folds} --")
        X_train_fold, X_val_fold = X_train_pool[train_idx], X_train_pool[val_idx]
        y_train_fold, y_val_fold = y_train_pool[train_idx], y_train_pool[val_idx]

        model = GMMClassifier(n_classes=len(LABEL_NAMES), n_components=2, covariance_type='full', random_state=42)
        
        model.fit(X_train_fold, y_train_fold)
        
        # No eval_set or evals_result for GMM, so skip learning curves
        y_pred_val = model.predict(X_val_fold)
        val_accuracy = np.mean(y_pred_val == y_val_fold)
        # Store history for this fold (dummy values for loss)
        cv_histories.append({
            'fold': fold, 'epoch': 1,
            'train_loss': 0,
            'train_accuracy': 0,
            'val_loss': 0,
            'val_accuracy': val_accuracy,
        })
        print(f"Fold {fold} completed. Final Val Accuracy: {val_accuracy:.4f}")

    # Save detailed CV history to CSV
    cv_df = pd.DataFrame(cv_histories)
    cv_df.to_csv(history_path, index=False)
    print(f"\nCross-validation history saved to {history_path}")

    # Final Model Training & Evaluation
    print("\n-- Training Final Model on Full Training Pool --")
    final_model = GMMClassifier(n_classes=len(LABEL_NAMES), n_components=2, covariance_type='full', random_state=42)
    final_model.fit(X_train_pool, y_train_pool)

    print("\n-- Evaluating Final Model on Held-Out Test Set --")
    if is_frame_scenario:
        y_pred_frames_final = final_model.predict(X_test_final)
        y_pred_final, y_true_final = vote_on_predictions(y_pred_frames_final, y_test_final, config)
    else:
        y_pred_final = final_model.predict(X_test_final)
        y_true_final = y_test_final

    elapsed_time = time.time() - start_time
    final_report = classification_report(y_true_final, y_pred_final, target_names=LABEL_NAMES, zero_division=0)
    with open(report_path, "w") as f:
        f.write(f"--- Final Test Report for {scenario_name} ---\n\n{final_report}")
        f.write(f"\nElapsed Time: {elapsed_time:.2f} seconds\n")
    
    print("\n--- Final Test Set Report ---")
    print(final_report)
    print(f"Final report saved to {report_path}")

# --- 5. Main Execution Block ---

if __name__ == "__main__":
    random.seed(42), np.random.seed(42)
    config = Config()
    AUDIO_DIR, LABEL_DIR = "./audio", "./strong_labels"
    
    all_segments = list(load_and_segment_audio(AUDIO_DIR, LABEL_DIR, config))
    all_segment_waveforms, all_segment_labels = [item[0] for item in all_segments], np.array([item[1] for item in all_segments])
    print(f"Loaded {len(all_segments)} total 10-second segments.")
    print(f"Segment labels: {Counter(all_segment_labels)}")

    # --- Scenario 1: Aggregated Features ---
    X_agg = np.array([extract_aggregated_segment_features(seg, config.sample_rate, config) for seg in tqdm(all_segment_waveforms, desc="Scenario 1: Aggregating features")])
    X_train_pool_agg, X_test_final_agg, y_train_pool_agg, y_test_final_agg = train_test_split(
        X_agg, all_segment_labels, test_size=config.test_split_ratio, random_state=42, stratify=all_segment_labels
    )
    run_evaluation_protocol(X_train_pool_agg, y_train_pool_agg, X_test_final_agg, y_test_final_agg, config,
                            "Aggregated Segment Features", config.agg_final_report_path, config.agg_cv_history_path)

    # --- Scenario 2: Frame-Level Features ---
    all_frame_features, all_frame_labels, segment_indices_for_frames = [], [], []
    for i, (segment_waveform, segment_label) in enumerate(tqdm(all_segments, desc="Scenario 2: Extracting frame features")):
        frame_features = extract_frame_level_features(segment_waveform, config.sample_rate, config)
        all_frame_features.extend(frame_features)
        all_frame_labels.extend([segment_label] * len(frame_features))
        segment_indices_for_frames.extend([i] * len(frame_features))
        
    X_frames, y_frames, segment_indices = np.array(all_frame_features), np.array(all_frame_labels), np.array(segment_indices_for_frames)
    
    # Split frame data based on their parent segment to prevent data leakage
    unique_segments = np.unique(segment_indices)
    train_segments, test_segments = train_test_split(unique_segments, test_size=config.test_split_ratio, random_state=42, stratify=all_segment_labels)
    
    train_mask, test_mask = np.isin(segment_indices, train_segments), np.isin(segment_indices, test_segments)
    X_train_pool_frames, X_test_final_frames = X_frames[train_mask], X_frames[test_mask]
    y_train_pool_frames, y_test_final_frames = y_frames[train_mask], y_frames[test_mask]

    run_evaluation_protocol(X_train_pool_frames, y_train_pool_frames, X_test_final_frames, y_test_final_frames, config,
                            "Frame-Level Features with Voting", config.voted_final_report_path, config.voted_cv_history_path, is_frame_scenario=True)
