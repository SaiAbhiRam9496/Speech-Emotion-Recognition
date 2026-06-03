# Speech Emotion Recognition (SER)

## Overview
This project recognizes human emotions from speech audio using deep learning. It identifies 6 emotions: neutral, happy, sad, angry, fear, and disgust.

## How It Works

### 1. Data Collection
Uses 4 public emotion datasets:
- **RAVDESS**: 1056 audio files from actors
- **TESS**: 2400 audio files from 2 female speakers
- **CREMA-D**: 7442 audio files from multiple speakers
- **SAVEE**: 360 audio files from 4 male speakers

**Total: 11,258 unique audio files** (augmented to 45,032 training samples)

### 2. Feature Extraction
Each audio file (2.5 seconds) is converted to **3,672 numerical features**:
- Zero-Crossing Rate (ZCR) — 108 values
- Root Mean Square Energy (RMSE) — 108 values
- MFCC (20 coefficients) — 2160 values
- Chroma features (12 pitch classes) — 1296 values

Data is split: **80% training, 10% validation, 10% testing**

### 3. Three Deep Learning Models

**Model A: 1D Convolutional Neural Network (CNN)**
- 5 convolutional blocks with pooling
- Fast training, baseline performance

**Model B: CNN + Bidirectional LSTM**
- Same CNN structure with LSTM layer
- Captures long-range dependencies in audio
- Higher complexity but WORSE accuracy than Model A (86.7% vs 90.59% — overfitting)

**Model C: Averaging Ensemble**
- Combines Model A and Model B
- Averages their predictions
- Middle performance: 90.0% (between CNN at 90.59% and BiLSTM at 86.7%)

### 4. Training Settings
- Batch size: 64
- Epochs: up to 100
- Optimizer: Adam (learning rate 0.001)
- Early stopping: if no improvement for 5 epochs

### 5. Results
All models evaluated on **speaker-independent test set**:

| Model | Accuracy | F1 Score | AUC-ROC | AUC-PRC |
|-------|----------|----------|---------|---------|
| CNN | 90.59% | 90.61% | 99.15% | 96.77% |
| CNN+BiLSTM | 86.7% | 86.71% | 98.47% | 94.33% |
| Ensemble | 90.0% | 90.01% | 98.99% | 96.22% |

### 6. Interpretability
Uses **LIME** (Local Interpretable Model-agnostic Explanations) to show which audio features influence emotion predictions.

## Files Saved
- `X_train_scaled.npy`, `X_val_scaled.npy`, `X_test_scaled.npy` — Processed data
- `cnn_model.keras`, `bilstm_model.keras`, `ensemble_model.keras` — Trained models
- `training_curves.png` — Accuracy and loss plots
- `confusion_matrices.png` — Prediction mistakes by emotion
- `lime_explanations.png` — Feature importance visualizations
- `results_summary.txt` — Final metrics

## Requirements
- Python 3, TensorFlow, Keras, NumPy, librosa
- GPU recommended (T4 GPU used in development)
- Google Drive for storage

## How to Use
1. Mount Google Drive
2. Upload datasets to `/content/drive/MyDrive/Deep Learning Project/Datasets/`
3. Run cells in order (1 → 37)
4. Models automatically train and save to Drive
5. View results in `results/` folder

## Key Insight
The simple **CNN model (Model A) achieves best accuracy (90.59%)** by avoiding overfitting. Adding BiLSTM increases complexity without improving test performance, proving that simpler architectures often generalize better on this task.
