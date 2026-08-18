# 🌌 Gravitational Wave Signal Classification with CNN

Deep learning approach for gravitational-wave signal detection using a
**Convolutional Neural Network (CNN)** trained on time-frequency spectrograms
of simulated detector data.

## Overview

This project investigates the use of deep learning for identifying
gravitational-wave signals embedded in noisy detector data.

A convolutional neural network is trained to perform binary classification
between:

* **Noise** — detector noise, including simulated transient glitches.
* **Signal + Noise** — gravitational-wave signals injected into detector noise.

The project follows a complete end-to-end pipeline, from physical waveform
generation and detector-noise simulation to spectrogram construction, CNN
training, performance evaluation, SNR-dependent sensitivity studies,
out-of-distribution generalization tests, and comparison with matched
filtering.

When PyCBC is available, physical gravitational-wave waveforms are generated
using waveform models such as **IMRPhenomD** and **TaylorT4**. A synthetic
chirp implementation is provided as a fallback when PyCBC is unavailable.

## Dataset

The dataset is generated programmatically rather than loaded from an
external database.

A total of **10,000 samples** are generated for the main dataset, containing
both noise-only and signal-plus-noise examples. The dataset is split into:

```text
Training:    70%
Validation:  15%
Test:        15%
```

The train/validation/test split is stratified and uses a fixed random seed for reproducibility.
The simulated detector data include:
Gaussian noise with variable amplitude;
transient sine-Gaussian glitches;
gravitational-wave signals with variable signal-to-noise ratio;
varying source distances;
varying low-frequency cutoffs;
different compact-object mass ranges;
aligned spin parameters.
The default sampling frequency is **2048 Hz**, with signal windows between
**1 and 2 seconds**.
All input spectrograms are normalized using the mean and standard deviation
computed exclusively from the training set, which prevents information from
the validation and test sets from entering the training preprocessing.

## 🧠 Methodology

### 1. Gravitational-Wave Signal Generation

When PyCBC is available, physical gravitational-wave waveforms are generated
using parameterized compact-binary systems.
The default parameter ranges include:

```text
Sampling frequency:       2048 Hz
Signal duration:          1–2 s
Signal SNR:               5–20
Distance:                 100–1000 Mpc
Low-frequency cutoff:     40–60 Hz
Aligned spin:             -0.8 to 0.8
```

The project supports different compact-object mass ranges corresponding to
binary neutron-star, binary black-hole, and neutron-star–black-hole systems.
When PyCBC is not available, the project falls back to synthetic quadratic
chirps generated with SciPy.

### 2. Detector Noise and Glitches
Background data are simulated using Gaussian noise with variable amplitude.
Transient glitches can also be injected into the detector background with a
default probability of:

```text
glitch probability = 0.50
```

The simulated glitches are sine-Gaussian transients with variable frequency,
quality factor, and SNR.
This provides a more challenging classification problem than simply
distinguishing clean gravitational-wave signals from stationary Gaussian
noise.

### 3. Time-Frequency Representation
The simulated time-series data are converted into **spectrograms** using a
Short-Time Fourier Transform (STFT).
Default spectrogram parameters:

```text
nperseg  = 64
noverlap = 32
```

The resulting time-frequency representation is used as the input to the
convolutional neural network.

### 4. CNN Architecture
The classifier is implemented using **PyTorch**.
The convolutional backbone consists of four convolutional blocks with
increasing numbers of filters:

```text
Input Spectrogram
       ↓
Conv2D (32)
BatchNorm + ReLU
MaxPool
       ↓
Conv2D (64)
BatchNorm + ReLU
MaxPool
       ↓
Conv2D (128)
BatchNorm + ReLU
MaxPool
       ↓
Conv2D (256)
BatchNorm + ReLU
       ↓
Global Average Pooling
       ↓
Fully Connected Layers
       ↓
Dropout
       ↓
Sigmoid
       ↓
P(signal)
```

The convolutional layers use kernel size 3 with padding 1.
Global Average Pooling converts the convolutional feature maps into a
fixed-size representation before the fully connected classifier.
The default dropout probabilities are:

```text
Dropout 1 = 0.50
Dropout 2 = 0.30
```

### 5. Training
The network is trained using:
**Adam** optimizer;
**Binary Cross-Entropy (BCE)** loss;
mini-batch training;
learning-rate reduction on validation-loss plateaus;
early stopping;
best-validation-model checkpointing.
Default training configuration:

```text
Batch size       = 64
Learning rate    = 5 × 10⁻⁵
Maximum epochs   = 14
Early stopping   = 10 epochs
```

The project also fixes the random seed to **42** where applicable to improve
reproducibility.

### 6. Performance Evaluation
The trained classifier is evaluated on an independent test set.
Performance is assessed using:
classification accuracy;
ROC curves;
ROC AUC;
confusion matrix;
detailed classification metrics;
accuracy as a function of signal-to-noise ratio.
The standard classification threshold is:

```text
P(signal) ≥ 0.5
```

for assigning an example to the signal-plus-noise class.

### 7. Sensitivity vs. SNR
The model's detection performance is investigated across different injected
signal-to-noise ratios.
For each SNR value, new noise realizations are generated independently and
the model's classification accuracy is measured.
This provides a direct view of how the detector performance changes as the
gravitational-wave signal becomes weaker relative to the detector noise.

### 8. Generalization to Unseen Parameters
A separate evaluation dataset is generated using parameter ranges that are
not used in the original training distribution.
The generalization study includes:
random component spins;
mixed gravitational-wave polarizations;
varying distances;
varying SNR;
modified waveform parameters.
The generalization dataset is **not used during training** and is intended
to evaluate out-of-distribution performance.
For the robustness study, the SNR range is extended to:
```text
SNR = 3–25
```
and the aligned spin range is extended to:
```text
spin = -0.95 to 0.95
```
with distances ranging from:
```text
50–1500 Mpc
```

### 9. Generalization to Unseen Signal Morphologies
The project also evaluates the CNN using signal morphologies that differ from
the chirp morphology used during training.
This test investigates whether the classifier learns more general
time-frequency characteristics rather than simply memorizing the specific
waveform morphology present in the training data.

### 10. Comparison with Matched Filtering
The final part of the project compares CNN-based detection with
**matched filtering**.
The comparison considers two complementary regimes:
- **Ideal conditions**, where the matched-filter template exactly matches
the injected waveform.
- **More realistic conditions**, involving non-ideal noise, glitches, and
waveform variability.
The comparison is not intended as a perfectly controlled benchmark because
the two approaches rely on different assumptions.
Matched filtering is theoretically optimal when the signal template and
noise model are known exactly, whereas the CNN learns its decision function
from the training distribution.

## Configuration
The main physical and machine-learning parameters are defined in a single
configuration dictionary:
```text
PARAMS = {
    # ── Time / sampling parameters ───────────────────────────
    "delta_t"       : 1.0 / 2048,   # Time step [s] (fs = 2048 Hz)
    "window_min"    : 1.0,           # Minimum window duration [s]
    "window_max"    : 2.0,           # Maximum window duration [s]

    # ── Detector noise ───────────────────────────────────────
    "sigma_min"     : 1e-23,         # Minimum Gaussian noise amplitude
    "sigma_max"     : 1e-21,         # Maximum Gaussian noise amplitude
    "glitch_prob"   : 0.50,          # Probability of adding a glitch to noise

    # ── Gravitational-wave signal ────────────────────────────
    "snr_min"       : 5.0,           # Minimum injected SNR
    "snr_max"       : 20.0,          # Maximum injected SNR
    "d_min"         : 100.0,         # Minimum distance [Mpc]
    "d_max"         : 1000.0,        # Maximum distance [Mpc]
    "f_lower_min"   : 40.0,          # Minimum low-frequency cutoff [Hz]
    "f_lower_max"   : 60.0,          # Maximum low-frequency cutoff [Hz]

    # ── Masses (system-dependent ranges) ─────────────────────
    "m_ns_min"      : 1.2,           # Minimum neutron star mass [M☉]
    "m_ns_max"      : 2.0,           # Maximum neutron star mass [M☉]
    "m_bh_min"      : 10.0,          # Minimum black hole mass [M☉]
    "m_bh_max"      : 50.0,          # Maximum black hole mass [M☉]
    "m_nsbh_bh_min" : 8.0,           # Minimum BH mass in NSBH systems [M☉]
    "m_nsbh_bh_max" : 30.0,          # Maximum BH mass in NSBH systems [M☉]

    # ── Spin ────────────────────────────────────────────────
    "spin_min"      : -0.8,          # Aligned spin z-component minimum (dimensionless)
    "spin_max"      :  0.8,          # Aligned spin z-component maximum (dimensionless)
    "use_spin"      : True,          # Enable/disable spin parameters

    # ── Glitches ─────────────────────────────────────────────
    "glitch_f_min"  : 30.0,          # Minimum glitch frequency [Hz]
    "glitch_f_max"  : 300.0,         # Maximum glitch frequency [Hz]
    "glitch_Q_min"  : 3.0,           # Minimum Q-factor (sine-Gaussian)
    "glitch_Q_max"  : 20.0,          # Maximum Q-factor (sine-Gaussian)
    "glitch_snr_min": 5.0,           # Minimum injected glitch SNR
    "glitch_snr_max": 20.0,          # Maximum injected glitch SNR

    # ── Spectrogram ─────────────────────────────────────────
    "nperseg"       : 64,            # Samples per STFT segment
    "noverlap"      : 32,            # Overlap between segments

    # ── Dataset ─────────────────────────────────────────────
    "n_samples"     : 10000,           # Training + validation + test samples
    "test_size"     : 0.15,          # Test split fraction
    "val_size"      : 0.15,          # Validation split fraction (of total)

    # ── Training ────────────────────────────────────────────
    "batch_size"    : 64,
    "lr"            : 5e-5,
    "patience"      : 10,            # Early stopping patience
    "dropout1"      : 0.50,          # Dropout first FC layer
    "dropout2"      : 0.30,          # Dropout second FC layer
}
```
This makes it possible to modify the physical simulation and machine-learning
configuration without changing the rest of the pipeline.

## Hardware Acceleration
PyTorch automatically selects the available computational device:
```text
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
```
Therefore, the project can run on:
NVIDIA GPUs through CUDA;
Apple Silicon GPUs through MPS;
CPU-only systems.

## 📊 Results

These results were obtained with n_samples = 10000.

Additional figures are included in the main code.

### Learning Curves

![Learning Curves](learningcurves.png)

### ROC Curve and Confusion Matrix

![ROC Curve & Confusion Matrix](roc_confusion.png)

### Accuracy vs SNR

![Accuracy vs SNR](acc_vs_snr.png)

The complete numerical results and additional visualizations are generated
by the notebook.
The analysis includes:
- training and validation loss
- training and validation accuracy
- test-set classification performance
- ROC AUC
- confusion matrix
- accuracy as a function of SNR
- out-of-distribution generalization
- performance on unseen signal morphologies
- comparison with matched filtering

## 🔬 Matched Filtering

The project also compares the CNN approach with matched filtering under
ideal and non-ideal conditions.

The comparison is intended to illustrate the different assumptions and
operating regimes of the two approaches rather than establish a perfectly
fair benchmark.

## Model Checkpoint
The trained model can be saved as a PyTorch checkpoint containing:
```text
model_state_dict
optimizer_state_dict
training history
model parameters
```
The default checkpoint filename is:
```text
grav_wave_cnn6.pt
```
The model architecture is reconstructed from the stored configuration before
loading the learned weights.

## Technologies

- Python 3
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch
- Torchvision
- PyCBC
- LALSuite
- Jupyter Notebook

## Author
**Mathias Rendón Fernández**

Physics student — Universidad Autónoma de Madrid
