# Child Mind Institute — Detect Sleep States (Kaggle Competition)

## Overview

The [Child Mind Institute - Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states) competition (2023) required detecting sleep onset and wakeup events from wrist-worn accelerometer data (anglez + enmo signals sampled every 5 seconds). Evaluated as event detection Average Precision across multiple time tolerances.

Kaggle profile: [illidan7](https://www.kaggle.com/illidan7)

## Approach

### 1. Data Pipeline at Scale

127M rows of training time series across 277 subjects. Used cuDF (GPU DataFrames) for loading, aggressive memory reduction (dtype downcasting), and per-series file splitting. 7-fold GroupKFold by series_id ensuring no subject data leakage.

### 2. Feature Engineering (121 Features)

Engineered rolling statistics (mean, max, std) of anglez/enmo and their first-order differences at 9 window sizes (5 seconds to 8 hours), plus time-of-day features. Later reduced to 9 core features after ablation. Built as a reusable Python module.

### 3. Target Representation Evolution

Progressed from binary sleep/awake labels to Gaussian-smoothed probability distributions centered on onset/wakeup events, improving gradient signal for sequence models.

### 4. Model Progression

- **LightGBM baseline**: Insufficient for temporal patterns
- **Bidirectional LSTM** (TensorFlow/Keras): Chunked 48-hour windows with custom LR scheduling
- **GRU-UNet hybrid**: Combined GRU recurrent cells with U-Net encoder-decoder structure using KL divergence loss, directly predicting event probability peaks

### 5. Post-Processing

Applied sigmoid activation, low-pass Butterworth filtering for signal smoothing, then local maxima detection (scipy argrelmax) for precise event localization. Enforced minimum sleep duration and activity tolerance constraints.

### 6. Ensemble

Weighted combination of GRU-UNet predictions with "Tubo" (a ResNet34-based 1D U-Net segmentation model), leveraging model diversity between recurrent and convolutional architectures.

## Repository Structure

```
├── eda/
│   ├── initial-exploration.ipynb                  # Data shape, memory challenges
│   └── deep-exploration.ipynb                     # Null handling, timestamps, ExtraTrees baseline
├── data-pipeline/
│   ├── generate-cv-folds.ipynb                    # 7-fold GroupKFold by series_id
│   ├── numpy-fold-conversion.ipynb                # cuDF → numpy with feature engineering
│   └── gaussian-target-features.ipynb             # Gaussian-smoothed targets + reduced features
├── training/
│   ├── lgbm-baseline.ipynb                        # LightGBM baseline (insufficient)
│   ├── lstm-training-tensorflow.ipynb             # Primary BiLSTM training (10 versions)
│   ├── lstm-training-with-validation.ipynb        # LSTM with event detection AP validation
│   ├── grunet-base-2feat.ipynb                    # ★ GRU-UNet with KL divergence loss
│   └── grunet-base-7feat.ipynb                    # GRU-UNet with expanded features
├── inference/
│   ├── grunet-inference.ipynb                     # GRU-UNet inference pipeline
│   └── postprocessing-peak-detection.ipynb        # ★ LPF + local maxima event detection
├── ensemble/
│   └── final-ensemble-submission.ipynb            # ★ GRU-UNet + Tubo ensemble (10 versions)
└── utils/
    └── feature-engineering-utils.py               # 121-feature engineering function
```

## Tech Stack

- **Models**: GRU-UNet (PyTorch), BiLSTM (TensorFlow/Keras), LightGBM
- **Data Processing**: cuDF/RAPIDS (GPU), scipy (signal processing)
- **Experiment Tracking**: Weights & Biases
- **Infrastructure**: Kaggle Notebooks (GPU, TPU)

## Competition

- **Name**: [Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)
- **Type**: Event detection (time series)
- **Metric**: Event detection Average Precision
- **Timeline**: September — December 2023
