# Exercise 3 — CNN Analysis on CIFAR-10 (From Scratch)

This repository contains the solution for **AI Exercise 3**: training a custom Convolutional Neural Network (CNN) on **CIFAR-10** and analyzing internal feature representations (feature maps) across layers.

> **Key point:** The model is **trained from scratch** (no pretrained/backbone models).

---

## Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Notes & Troubleshooting](#notes--troubleshooting)

---

## Project Overview

The goals are to:

1. Build and train a CNN classifier on CIFAR-10 (10 classes).
2. Track training/validation metrics (loss and accuracy).
3. Visualize learned representations by extracting and plotting **feature maps** from early, mid, and deep layers.

Interpretation (high level):

- **Early layers** learn edges and simple color/texture primitives.
- **Mid layers** learn more structured patterns (corners, repeated textures).
- **Deep layers** encode more class-relevant abstractions.

---

## Dataset

- **CIFAR-10**: 60,000 color images (32×32) in 10 classes
  - 50,000 train / 10,000 test

### Preprocessing & Augmentation

**Training transforms**:

- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`
- `Normalize(mean, std)`

**Validation transforms**:

- `Normalize(mean, std)` only

Normalization parameters (standard CIFAR-10):

- `mean = (0.4914, 0.4822, 0.4465)`
- `std  = (0.2470, 0.2435, 0.2616)`

---

## Approach

### Model: VGG-style Custom CNN (8 Conv Layers)

The network uses **4 convolutional blocks**, each block:

- Conv → BatchNorm → ReLU
- Conv → BatchNorm → ReLU
- MaxPool

Then:

- Global Average Pooling (GAP)
- Fully-connected classifier head (10 classes)

This is intentionally **hand-built** to study how representations evolve with depth.

### Optimization

- Optimizer: **SGD + Momentum**
- Weight decay: **5e-4**
- Learning rate schedule: **Cosine Annealing**
- Batch size: **128**
- Epochs: **50**

---

## Results

A representative run achieved:

- **Best Validation Accuracy:** **~92–93%** on CIFAR-10

> Results may vary slightly due to randomness (shuffling + augmentation).  
> See the notebook outputs and generated plots for run details.

---

## Repository Structure

```
AI_Exercise3_YourName/
├── CNN_Analysis.ipynb        # Main notebook (training + analysis)
├── Report.pdf                # Report (if required separately)
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── data/                     # CIFAR-10 downloaded here by torchvision
└── outputs/
    ├── figures/              # Plots & feature map images
    └── models/               # Saved weights (best_model.pth)
```

---

## How to Run

### 1) Create an environment (recommended)

#### Windows (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Launch the notebook

```bash
jupyter notebook
```

Open:

- `CNN_Analysis.ipynb`

### 3) Training & analysis

The notebook will:

- download CIFAR-10 to `./data`
- train the model
- save the **best checkpoint** to `outputs/models/best_model.pth`
- save plots/feature maps under `outputs/figures/`

---

## Outputs

After running the notebook you should have:

### Metrics

- Training/validation **loss curve**
- Training/validation **accuracy curve**

### Model checkpoint

- `outputs/models/best_model.pth` (best validation accuracy)

### Feature map visualizations

Saved images that illustrate activations across:

- early convolutional layers (edge/texture detectors)
- mid layers (pattern combinations)
- deeper layers (more abstract class-driven features)

---

## Notes & Troubleshooting

### Windows DataLoader workers

If you see DataLoader-related issues on Windows, keep:

- `num_workers=0` (safe default)

### GPU vs CPU

The code automatically uses CUDA if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Reproducibility (optional)

For more repeatable results, you can fix seeds in the notebook:

- `torch.manual_seed(...)`
- `torch.cuda.manual_seed_all(...)`

---

## Credits

- CIFAR-10 is provided through `torchvision.datasets.CIFAR10`.
