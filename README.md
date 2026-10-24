# NP-SleepNet

**When Neyman–Pearson Meets Diffusion in Multimodal Sleep Stage Classification**

NP-SleepNet is a deep learning framework for automatic sleep stage classification, designed to address the persistently **low recognition rate of the N1 stage**. The project consists of two core components:

1. **NP classification framework** — based on the Neyman–Pearson (NP) criterion, it explicitly controls the False Negative Rate (FNR) of the N1 class at a user-specified target risk level α, instead of only optimizing overall accuracy.
2. **SleepDiffu** — a diffusion model tailored to the N1 class, used to synthesize high-fidelity N1 samples. These synthetic samples replace the real samples that NP calibration would otherwise need to hold out, alleviating the training-data loss caused by N1 scarcity.

---

## Table of Contents

- [Background & Motivation](#background--motivation)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [1. Baseline / NP Classification Training](#1-baseline--np-classification-training)
  - [2. SleepDiffu Training & Sample Generation](#2-sleepdiffu-training--sample-generation)
  - [3. Data Augmentation (augment.py)](#3-data-augmentation-augmentpy)
- [Code Overview](#code-overview)
- [Citation](#citation)

---

## Background & Motivation

Existing sleep stage classification methods achieve an overall accuracy of roughly 85%, yet the recall of the **N1 stage** — the transitional stage between wakefulness and sleep — has remained around 50%. Reliable N1 identification is critical for diagnosing sleep disorders. To address this, this project:

- Adopts a Neyman–Pearson classification framework that keeps the N1 FNR under a user-specified target risk α;
- Notes that NP calibration requires holding out an independent set of real N1 samples. Since N1 makes up only 6.3% of the Sleep-EDF-20 dataset, reserving real samples for calibration substantially shrinks the effective training set and hurts overall accuracy;
- Proposes SleepDiffu, a diffusion-based generative model that synthesizes N1 samples for calibration, preserving the rigorous FNR guarantee of NP classification while mitigating the N1 data-scarcity problem.

## Key Results

At a target risk of α = 0.2 (Sleep-EDF-20 dataset):

| Method | N1 FNR | Overall Accuracy |
|---|---|---|
| Standard classification (no NP constraint) | 53.2% | 86.3% |
| NP classification (real-sample calibration) | 19.5% | 78.3% |
| NP classification + SleepDiffu (synthetic-sample calibration) | ~20% (α maintained) | **81.2%** (+2.9% over real-sample calibration) |

## Repository Structure

```
NPSleepNet/
├── model.py            # SleepStageModel: residual CNN blocks + bidirectional GRU + Transformer encoder
├── EEG_data.py          # Data loading, multi-channel tensor dataset construction, sample visualization
├── utils_eeg.py         # Subject/sequence ID loading, per-class sample counts, Focal Loss, etc.
├── sleepstage.py        # AASM sleep stage label definitions (W/N1/N2/N3/REM/MOVE/UNK)
├── logger.py            # Logging utility (file + console)
├── augment.py           # N1-targeted data augmentation for 3-channel signals (noise/shift/scale/filter/warp/etc.)
├── diffusion.py          # SleepDiffu diffusion model (DDPM/DDIM denoiser, training & generation, quality validation)
├── diffusion1.py         # Earlier / alternative implementation of SleepDiffu
├── trainer.py             # Single-model training and NP calibration (calibrate_threshold / np_predict)
├── train.py / train1.py   # Single-model NP classification training scripts (20-fold leave-subjects-out CV)
├── train2.py              # Multi-model ensemble NP classification training script
├── train3.py              # Multi-model ensemble NP classification training script (alternate configuration)
└── train_diffu.py / train_diffu1.py  # N1-balancing training scripts using SleepDiffu
```

## Requirements

The project is implemented in PyTorch. Main dependencies (adjust version pins to match your local CUDA setup):

```
python >= 3.9
torch >= 1.13
numpy
scikit-learn
matplotlib
scipy        # used by validate_generation() in diffusion.py (scipy.signal.welch for spectral consistency checks)
```

We recommend using a dedicated conda / venv environment and installing the PyTorch build that matches your GPU/CUDA version.

## Data Preparation

This project is built on the **Sleep-EDF (Expanded)** dataset (EDF-20 / EDF-78 subsets). Raw PSG recordings must be preprocessed into 30-second-epoch `.npz` files, each containing:

- `x`: the signal array (single channel, length 3000, corresponding to 100 Hz × 30 s)
- `y`: the corresponding sleep stage label (integer, see `sleepstage.py`)
- `fs`: the sampling rate

Data should be organized by channel into three separate directories, with a text file listing the subject IDs used for cross-validation:

```
<dataset_root>/
├── fpz_cz/          # Fpz-Cz EEG channel, *.npz
├── pz_oz/           # Pz-Oz EEG channel, *.npz
├── eog/             # EOG signal, *.npz
└── sleepedf.txt      # subject IDs used in cross-validation, one integer per line
```

> **Note**: Scripts such as `train1.py` / `train2.py` currently use hard-coded local/server data paths (e.g. `D:/DL_datasets/...` or `/bigdata/statsdept/...`). Update these to relative paths or externalize them via a config file before reuse.

## Usage

### 1. Baseline / NP Classification Training

`train1.py` (single model) and `train2.py` / `train3.py` (model ensembles) implement a 20-fold leave-subjects-out cross-validation pipeline: train `SleepStageModel` → calibrate the N1 decision threshold on held-out N1 samples via NP calibration → predict on the test set under the NP rule and report the confusion matrix / per-class accuracy.

```bash
python train.py
```

Training logs (per-fold accuracy, confusion matrix, per-class accuracy) are written to console and log file via `logger.py`.

The core NP logic lives in `trainer.py`:

- `calibrate_threshold(model, calib_loader, alpha, n1_label)`: computes the N1 decision threshold on the calibration set for a target risk α;
- `np_predict(model, loader, threshold, n1_label)`: applies the calibrated threshold to prioritize N1 predictions, falling back to the argmax softmax class for other samples.

### 2. SleepDiffu Training & Sample Generation

The `Sleepdiffu` class in `diffusion.py` implements a conditional diffusion model over 3-channel (Fpz-Cz, Pz-Oz, EOG) N1 signals (DDPM forward noising with optional DDIM sampling, EMA weights, per-sample amplitude bootstrapping, etc.).

```bash
python train_diffu.py    # on the EDF-20 subset
python train_diffu1.py   # on the EDF-78 subset
```

Core interfaces:

- `Sleepdiffu.fit(x_n1)`: trains the denoiser on real N1 samples;
- `Sleepdiffu.generate(n_samples)`: samples the specified number of synthetic N1 samples;
- `validate_generation(x_real, x_fake)`: checks consistency between real and generated samples (mean/std, power spectral density correlation, etc.);
- `balance_with_diffu(train_dataset, device, label_n1)`: trains SleepDiffu and uses the generated samples to balance the N1 class for downstream NP calibration.

### 3. Data Augmentation (augment.py)

`augment_N1` applies data augmentation to 3-channel N1 signals by randomly selecting two augmentation types from `{noise, shift, scale, filter, warp, freq, crop}`, applied to a single randomly chosen channel per sample — preserving the discriminative differences between channels. It can be used as a standard augmentation strategy alongside SleepDiffu-generated samples.

## Code Overview

- **Model architecture (`model.py`)**: `SleepStageModel` extracts local temporal features with residual convolutional blocks (`ResidualBlock`), models intra-epoch sequential dependencies with a bidirectional GRU, fuses global context with a Transformer encoder, and outputs a 5-class prediction (W/N1/N2/N3/REM) through a fully connected head.
- **Label scheme (`sleepstage.py`)**: follows the AASM sleep staging standard — `W=0, N1=1, N2=2, N3=3, REM=4` (plus `MOVE` and `UNK` for filtering invalid/unknown segments during preprocessing).
- **Multimodal dataset construction (`EEG_data.py`)**: `create_multi_tensor_dataset` stacks the Fpz-Cz, Pz-Oz, and EOG signals into a `(N, 3, L)` tensor; `visualize_gan_samples` provides side-by-side visual comparisons of generated vs. real samples.

## Citation

If you find this project useful for your research, please cite:

```bibtex
@article{npsleepnet,
  title   = {NP-SleepNet: When Neyman--Pearson Meets Diffusion in Multimodal Sleep Stage Classification},
  author  = {TODO: add authors},
  journal = {TODO: add venue},
  year    = {TODO}
}
```
