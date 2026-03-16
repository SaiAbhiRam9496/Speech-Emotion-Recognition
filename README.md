# Speech Emotion Recognition (SER) using Deep Learning

A multi-dataset approach combining four English speech emotion datasets with hand-crafted acoustic features and deep learning models to achieve robust emotion classification.

---

## Project Overview

This project develops a Speech Emotion Recognition system that identifies human emotions from speech signals using 4 complementary datasets, 4 acoustic features, and 3 deep learning models. The ensemble approach achieves 76% test accuracy, outperforming individual models.

Repository: https://github.com/SaiAbhiRam9496/Speech-Emotion-Recognition

---

## Datasets

Four English-language speech emotion datasets combined in this project:

### 1. RAVDESS (Ryerson Audio-Visual Emotion Database)
- 1,440 audio files from 24 professional actors (12M, 12F)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- High-quality studio recordings with balanced phonetics and 2 intensity levels per emotion

### 2. TESS (Toronto Emotional Speech Set)
- 2,800 audio files from 2 female actors (ages 26 & 64)
- 7 emotions: anger, disgust, fear, happiness, neutral, sadness, surprise
- Canadian English with clear articulation and natural speech patterns

### 3. SAVEE (Surrey Audio-Visual Expressed Emotion)
- 480 audio files from 4 male actors
- 7 emotions: anger, disgust, fear, happiness, neutral, sadness, surprise
- British English with video-synchronized recordings and consistent emotional expression

### 4. CREMA-D (Crowdsourced Emotional Multimodal Actors Database)
- 7,442 audio files from 91 diverse actors (48M, 43F with varied ethnicities)
- 6 emotions: anger, disgust, fear, happiness, neutral, sadness
- Large-scale database with crowdsourced labels and multiple emotion intensities

Combined Statistics: ~12,000 utterances | 118+ speakers | 6 unified emotions | ~20+ hours audio | 16 kHz sample rate

---

## Feature Extraction

Four acoustic features extracted from preprocessed audio signals:

### 1. MFCC (Mel-Frequency Cepstral Coefficients)
- Captures spectral characteristics by converting to mel-scale and applying DCT
- 13-40 coefficients per frame to encode emotional speech patterns (vibrato, nasalization, pitch variations)

### 2. Zero Crossing Rate (ZCR)
- Measures frequency content by counting signal zero-crossings
- High ZCR in excited/angry speech; low ZCR in calm/sad emotions

### 3. Root Mean Square Energy (RMSE)
- Represents loudness/energy levels over time
- Anger/happiness show high energy; sadness/neutral show lower energy

### 4. Chroma STFT (Chroma Short-Time Fourier Transform)
- Energy distribution across 12 pitch classes (musical notes)
- Captures pitch stability and vibrato patterns independent of absolute pitch

Final Feature Vector: ~100-150 total features with mean, std, min, max, median, kurtosis, and skewness statistics

---

## Model Architecture

---

### Model 1: CNN (Convolutional Neural Network)

Fully-connected neural network for capturing non-linear feature interactions.

| Layer     | Configuration                                 |
|           |                                               |
| Input     | Feature Vector                                |
| Dense     | 128 units, ReLU + BatchNorm + Dropout(0.3)    |
| Dense     | 64 units, ReLU + BatchNorm + Dropout(0.3)     |
| Dense     | 32 units, ReLU + Dropout(0.2)                 |
| Output    | 6 units (emotions), Softmax                   |

Training Configuration:

| Parameter         | Value                     |
|                   |                           |
| Optimizer         | Adam (lr=0.001)           |
| Loss Function     | Categorical Cross-Entropy |
| Batch Size        | 32                        |
| Epochs            | 100-200 (Early Stopping)  |
| Total Parameters  | ~40-50K                   |

---

### Model 2: BiLSTM (Bidirectional LSTM)

Recurrent neural network capturing temporal dependencies in sequential audio features.

| Layer     | Configuration                                     |
|           |                                                   |
| Input     | Feature Sequence                                  |
| Bi-LSTM   | 128 units, return_sequences=True + Dropout(0.3)   |
| Bi-LSTM   | 64 units, return_sequences=False + Dropout(0.3)   |
| Dense     | 32 units, ReLU + Dropout(0.2)                     |   
| Output    | 6 units (emotions), Softmax                       |

Training Configuration:

| Parameter         | Value                         |
|                   |                               |
| Optimizer         | Adam (lr=0.001)               |
| Loss Function     | Categorical Cross-Entropy     |
| Batch Size        | 32                            |
| Epochs            | 100-200 (Early Stopping: 20)  |
| Total Parameters  | ~40-50K                        |

---

### Model 3: Ensemble (CNN + BiLSTM)

Combines predictions from both CNN and BiLSTM through soft voting for improved accuracy and robustness.

| Component         | Details                                   |
|                   |                                           |
| CNN Output        | 6-dimensional probability vector          |
| BiLSTM Output     | 6-dimensional probability vector          |
| Fusion Strategy   | Soft Voting (Average Probabilities)       |
| Final Prediction  | argmax(averaged probabilities)            |
| Weighting         | Proportional to individual model accuracy |

Advantages: Combines temporal (BiLSTM) and feature-based (CNN) learning for improved generalization and reduced variance.

---

## Results & Performance

### Model Accuracy

|     Model    | Training | Validation | Test     |
|:------------:|:--------:|:----------:|:--------:|
|     CNN      |   ~85%   |    75%     |  74%     |
|    BiLSTM    |   ~83%   |    74%     |  72%     |
|  Ensemble    |   ~84%   |    77%     |  76%     |

### Per-Emotion Metrics

|   Emotion  | CNN Prec | CNN Rec | BiLSTM Prec | BiLSTM Rec | Ensemble F1 |
|:----------:|:--------:|:-------:|:-----------:|:----------:|:-----------:|
|   Anger    |  0.78    |  0.76   |    0.75     |   0.73     |    0.78     |
|   Disgust  |  0.72    |  0.70   |    0.71     |   0.69     |    0.74     |
|   Fear     |  0.68    |  0.66   |    0.70     |   0.72     |    0.71     |
|  Happiness |  0.82    |  0.80   |    0.79     |   0.78     |    0.83     |
|   Neutral  |  0.75    |  0.76   |    0.74     |   0.75     |    0.77     |
|  Sadness   |  0.76    |  0.74   |    0.73     |   0.71     |    0.76     |

Key Finding: Ensemble model achieves 76% accuracy | Happiness & Anger best recognized | Disgust & Fear most confused

---

## Installation & Usage

Open the notebook directly in Google Colab for execution. This project requires GPU acceleration for optimal performance and is designed to run exclusively on Colab - no local Jupyter setup is available.

Run on Colab:
1. Visit [Google Colab](https://colab.research.google.com)
2. Upload or open the notebook: `Deep Learning SER - Best (74,72,76).ipynb`
3. Mount Google Drive for dataset access (if datasets stored in Drive)
4. Execute cells sequentially for preprocessing, training, and evaluation

Dataset Download Links (for Colab):
- [RAVDESS](https://zenodo.org/record/1188976)
- [TESS](https://tspace.library.utoronto.ca/handle/1807/24612)
- [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

Download the Project from : 
https://github.com/SaiAbhiRam9496/Speech-Emotion-Recognition.git

```

Install Dependencies (in Colab cell):
```bash
!pip install librosa numpy pandas scipy tensorflow scikit-learn matplotlib seaborn
```

---

## References

1. Primary: Nature (2025) - Speech Emotion Recognition using Multi-Dataset Approach  
   DOI: https://doi.org/10.1038/s41598-025-95734-z

2. Datasets:
   - Livingstone & Russo (2018) - RAVDESS: PLoS ONE, 13(5)
   - Pichora-Fuller et al. (2020) - TESS: Frontiers in Psychology, 11
   - Jackson & Haq (2014) - SAVEE: University of Surrey
   - Cao et al. (2014) - CREMA-D: IEEE TAC, 5(4)

3. Methods:
   - Ellis (2005) - PLP and RASTA analysis
   - McFee et al. (2015) - librosa: Audio analysis in Python
   - Goodfellow et al. (2016) - Deep Learning fundamentals
   - Hochreiter & Schmidhuber (1997) - LSTM architecture

---

Last Updated: March 2026 | Status: Active Development | License: MIT
