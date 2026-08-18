# 🌌 Gravitational Wave Signal Classification with CNN

Deep learning approach for gravitational-wave signal detection using a
Convolutional Neural Network (CNN) trained on time-frequency spectrograms.

## 📌 Project Overview

This project explores the use of deep learning for gravitational-wave
signal detection in noisy detector data.

The pipeline includes:

- Gravitational-wave signal generation
- Noise and glitch generation
- Time-frequency spectrogram construction
- CNN-based binary classification
- Model training with early stopping
- ROC and confusion-matrix evaluation
- Sensitivity analysis as a function of SNR
- Generalization to previously unseen signal parameters and morphologies
- Comparison with matched filtering

## 🧠 Methodology

The input time series are transformed into spectrograms, which are then
provided to a Convolutional Neural Network.

The project uses PyTorch for the neural network and PyCBC for the
generation of physical gravitational-wave waveforms when available.

## 📊 Results

### Learning Curves

![Learning Curves](figures /learning_curves.png)

![Learning Curves](CUAM/GitHub/Proyectos_Python/learningcurves.png)


### ROC Curve

![ROC Curve](figures/roc_curve.png)

### Confusion Matrix

![Confusion Matrix](figures/confusion_matrix.png)

### Accuracy vs SNR

![Accuracy vs SNR](figures/accuracy_vs_snr.png)

### Generalization

![Generalization](figures/generalization.png)

## 🔬 Matched Filtering

The project also compares the CNN approach with matched filtering under
ideal and non-ideal conditions.

The comparison is intended to illustrate the different assumptions and
operating regimes of the two approaches rather than establish a perfectly
fair benchmark.

## 🛠️ Requirements

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
