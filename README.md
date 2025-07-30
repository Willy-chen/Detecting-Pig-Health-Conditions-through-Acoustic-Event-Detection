# Detecting-Pig-Health-Conditions-through-Acoustic-Event-Detection
Classification of pig breathing sounds using prototypical network with k-means clustering and query attention

## Overview

This project develops an automated system to classify pig breathing sounds for respiratory health monitoring using advanced machine learning techniques. The system processes audio recordings captured through stethoscopes placed on pigs' backs and distinguishes between different respiratory patterns including normal breathing, light breathing, heavy breathing, and heavy breathing with noise.

## Methodology

The approach combines traditional machine learning models with novel deep learning architectures:
1. Feature Extraction
- Traditional Models: 76 acoustic features per frame (MFCC, Log Mel Spectrogram, Chroma, Spectral features, etc.)
- Deep Learning Models: Mel spectrogram analysis with pretrained encoders (VGGish, Whisper, AST)

2. Novel Architectures
- K-means Prototypical Network (KPN): Uses multiple prototypes per class via k-means clustering to handle intra-class variance
- Query Attention K-means Prototypical Network (KAPN): Adds query-dependent attention mechanism to dynamically weight prototypes for each classification decision

3. Model Categories
- Traditional ML: GMM, HMM, XGBoost
- Deep Learning Baselines: CNN, CRNN, Transformer architectures
- Pretrained Encoders: VGGish, Whisper (tiny/base/small), Audio Spectrogram Transformer (AST)

## Dataset

- 42 audio files (~62 hours total) of pig breathing sounds
- 12 strongly labeled files (10.5 hours) with precise annotations
- 5 categories: breathing, light breathing, heavy breathing, heavy breathing with noise, no breathing
- Recorded using electronic stethoscopes in real farm environments
Note: Since the dataset is too large, and is provided by NPUST Animal Nutrigenomics Lab, the audio folder is not included in this repository 

# Setup

Download the required python packages:
```=python
pip install -r requirements.txt
```
Note: Not all packages in requirements.txt may be required.

# How to run

## Individual Experiments
Run specific experiment files:
```=python
python3 cv_test_frame_segment_gmm.py        # GMM baseline
python3 cv_test_frame_segment_hmm.py        # HMM baseline  
python3 cv_test_frame_segment_xgboost.py    # XGBoost baseline
python3 encoder-classifier.py               # Deep learning baselines
python3 encoder-proto.py                    # Standard prototypical network
python3 encoder-proto-kmeans.py             # K-means prototypical network
python3 encoder-proto-kmeans_attention.py   # Query attention K-means prototypical network
```
## Batch Experiments
Run multiple experiments at once:
```=shell
bash run.sh
```
## Configuration
- Adjust experiment parameters by modifying the configuration variables at the beginning of each Python file.
- Results will be output to the current working directory.

# Key Results
- Best Overall Performance: AST encoder + Linear classifier (82.58% accuracy, 53.74% macro F1)
- Novel Contribution: KAPN successfully stabilizes multi-prototype systems, preventing performance degradation seen in naive k-means approaches
- Traditional vs Deep Learning: Deep learning models significantly outperform traditional ML approaches on this complex acoustic task

# Repository Structure
- clean_label/, strong_labels/: Dataset and annotation files
- old_code/: Previous implementation versions
- cv_test_*.py: Traditional machine learning baselines
- encoder-*.py: Deep learning experiments with various architectures
- observe.ipynb: Data analysis and visualization notebook
- run.sh: Batch experiment runner

# Citation
Based on the master's thesis "Detecting Pig Health Conditions through Acoustic Event Detection" by Chen, Wei-Yu, National Taiwan University, 2025.
