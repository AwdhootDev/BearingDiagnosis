# Kurtosis — Bearing Fault Diagnosis Platform

AI-powered vibration signal analysis for automated bearing condition monitoring and fault classification.

## Overview

Kurtosis is an interactive Streamlit web application that diagnoses bearing faults from raw vibration signals. It combines **machine learning** and **deep learning** models in an ensemble pipeline, aggregating predictions across seven (or eight) classifiers to produce a robust final diagnosis.

Users upload a MATLAB (`.mat`) file containing vibration data (from datasets such as **CWRU** or **MFPT**), and the system automatically:

1. Loads and parses the vibration signal  
2. Segments it into fixed-size windows  
3. Extracts time-domain and frequency-domain features  
4. Generates Gramian Angular Field (GAF) image representations  
5. Runs inference across all models  
6. Aggregates predictions via majority voting to produce a final diagnosis  

### Fault Classes

| Class | Description |
|---|---|
| Normal | Healthy bearing — no fault detected |
| Ball Fault | Defect on the rolling element (ball) |
| Inner Race | Defect on the inner race of the bearing |
| Outer Race | Defect on the outer race of the bearing |

---

## Models

### Machine Learning Models

| Model | Input | Library |
|---|---|---|
| **XGBoost** | Time + frequency features | scikit-learn / XGBoost |
| **Random Forest** | Time + frequency features | scikit-learn |
| **SVM** | Scaled time + frequency features | scikit-learn |

### Deep Learning Models

| Model | Architecture | Input |
|---|---|---|
| **1D CNN** | 5-layer 1D convolutions with Tanh activations | Raw 1D signal windows (2048 samples) |
| **MSDC-DenseTCN** | Multi-Scale Dilated Causal convolutions + Dense Temporal Convolutional Network | Raw 1D signal windows (2048 samples) |
| **LSTM** | Stacked 2-layer LSTM | Raw 1D signal windows (2048 samples) |
| **TCN + Attention** | Temporal Convolutional Network with a learned attention mechanism | Raw 1D signal windows (2048 samples) |
| **2D CNN** | 3-layer 2D convolutions with BatchNorm | GASF + GADF images (224×224, 2-channel) |

---

## Preprocessing Pipeline

```
Raw .mat file
  → Signal extraction (CWRU DE_time / MFPT bearing format)
  → Windowing (2048-sample and 1024-sample non-overlapping windows)
  → Feature extraction (time-domain + frequency-domain)
  → GAF image generation (GASF & GADF for 2D CNN)
  → Model inference
  → Majority-vote aggregation → Final Diagnosis
```

### Time-Domain Features
Mean, Standard Deviation, RMS, Kurtosis, Peak-to-Peak, Skewness, Crest Factor, Shape Factor, Impulse Factor, Clearance Factor

### Frequency-Domain Features
Dominant Frequency, Spectral Energy, Spectral Centroid, Spectral Entropy, Spectral Bandwidth, Peak Amplitude

### Image Representations (for 2D CNN)
- **GASF** — Gramian Angular Summation Field  
- **GADF** — Gramian Angular Difference Field  

Both are generated from min-max normalized signal windows, converted to polar coordinates, and resized to 224×224 grayscale images.

---

## Project Structure

```
BearingFault/
├── app.py                          # Streamlit app (7-model ensemble)
├── app1.py                         # Streamlit app (8-model ensemble, includes 1D CNN)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── preprocessing/                  # Signal preprocessing modules
│   ├── MatLoader.py                # Loads .mat files (CWRU & MFPT formats)
│   ├── windowing.py                # Segments signal into fixed-size windows
│   ├── time_features.py            # Extracts time-domain statistical features
│   ├── frequency_features.py       # Extracts frequency-domain spectral features
│   └── image_processing.py         # Generates GASF/GADF images for 2D CNN
│
└── models/                         # Model architectures & pretrained weights
    ├── CNN_1D.py                   # 1D CNN architecture (FaultDetectionCNN)
    ├── CNN_2D.py                   # 2D CNN architecture (FaultDiagnosis)
    ├── LSTM.py                     # Stacked LSTM architecture (BearingLSTM)
    ├── msdc_dense.py               # MSDC-DenseTCN architecture (MSDCDenseTCN)
    ├── TCN_ATTENTION.py            # TCN + Attention architecture (TCNAttention)
    ├── xgb_combined_model.joblib   # Pretrained XGBoost weights
    ├── rf_combined_model.joblib    # Pretrained Random Forest weights
    ├── svm_combined_model.joblib   # Pretrained SVM weights
    ├── svm_scaler.joblib           # StandardScaler for SVM input
    ├── mixed_fault_detection_1dcnn.pth          # 1D CNN weights
    ├── best_mixed_msdc_densetcn (1).pth         # MSDC-DenseTCN weights
    ├── Raw_2channel_best_model.pth              # 2D CNN weights
    ├── Raw_CWRU_MFPT_best_LSTM_weights.pth      # LSTM weights
    └── Raw_CWRU_MFPT_best_TCN&Attention_weights.pth  # TCN+Attention weights
```

---

## Installation

### Prerequisites

- Python 3.9 or above

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd BearingFault
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the Streamlit app:

```bash
# 7-model ensemble (without 1D CNN)
streamlit run app.py

# 8-model ensemble (with 1D CNN)
streamlit run app1.py
```

Then open the URL shown in the terminal (default: `http://localhost:8501`).

### Steps

1. Upload a `.mat` file containing a bearing vibration signal (CWRU or MFPT format).
2. View the signal visualization and extracted features.
3. Click **🚀 Analyze Signal** to run inference across all models.
4. Review per-model predictions and the aggregated **Final Diagnosis**.

---

## Supported Datasets

| Dataset | Sampling Rate | Signal Key |
|---|---|---|
| [CWRU Bearing Data](https://engineering.case.edu/bearingdatacenter) | 12 kHz | `*_DE_time` |
| [MFPT Bearing Data](https://www.mfpt.org/fault-data-sets/) | Variable | `bearing.gs` |

---

## License

This project is for academic and educational purposes.
