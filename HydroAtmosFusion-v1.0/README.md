# HydroAtmosFusion (v1.0) & OceanReasonNet (v0.1)

**Multimodal Deep Learning for Extreme Rainfall Prediction Using Ocean-Atmosphere Coupling**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)

## Overview

HydroAtmosFusion is a dual-encoder multimodal deep learning framework that jointly processes ocean state variables (SST, SSH, EKE) and atmospheric reanalysis fields (CAPE, TCWV, wind) through **bidirectional cross-modal attention** for extreme rainfall prediction over the Maritime Continent.

This repository accompanies the manuscript submitted to *Geoscientific Model Development* (GMD).

### Key Features

- **Dual-encoder architecture**: ResNet-18 + SE attention for ocean; Swin-Transformer + ConvGRU for atmosphere
- **Bidirectional cross-modal attention**: Symmetric ocean-atmosphere information exchange (8-head, d=32)
- **Composite loss function**: Weighted MSE + Focal-CSI + Fractions Skill Score for extreme events
- **Physics-consistent synthetic data**: Deterministic generation following Clausius-Clapeyron, geostrophic balance, and MSE budget
- **Comprehensive evaluation**: RMSE, MAE, correlation, CSI99, EDI99, FSS99

## Installation

```bash
git clone https://github.com/jogipanggabean/HydroAtmosFusion.git
cd HydroAtmosFusion
pip install -r requirements.txt
pip install -e .
```

**Requirements**: Python >= 3.10, CUDA-compatible GPU with >= 16 GB memory (recommended)

## Quick Start

### 1. Generate Synthetic Dataset

```bash
python scripts/generate_synthetic.py --output data/synthetic_ocean_atmos.npz
```

Generates a 64x64 grid, 365-day physics-consistent dataset (100-130 E, 10S-10N).

### 2. Train the Model

```bash
python scripts/train.py \
    --data_path data/synthetic_ocean_atmos.npz \
    --output_dir checkpoints/ \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4
```

### 3. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data_path data/synthetic_ocean_atmos.npz \
    --output results.json
```

### 4. Generate Publication Figures

```bash
python scripts/generate_figures.py --output_dir figures/
```

## Repository Structure

```
HydroAtmosFusion-v1.0/
|-- hydroatmosfusion/          # Core Python package
|   |-- __init__.py
|   |-- models.py              # Model architectures (Eq. 17-21)
|   |-- losses.py              # Composite loss function (Eq. 22-25)
|   |-- metrics.py             # Evaluation metrics (Eq. 31-32)
|   |-- dataset.py             # Dataset and dataloaders
|-- scripts/
|   |-- generate_synthetic.py  # Synthetic data generation (Eq. 12-16)
|   |-- train.py               # Training pipeline
|   |-- evaluate.py            # Evaluation and ablation
|   |-- generate_figures.py    # Publication figures
|-- configs/
|   |-- default.yaml           # Default hyperparameters
|-- data/                      # Generated datasets (not tracked)
|-- figures/                   # Generated figures
|-- notebooks/                 # Jupyter notebooks (optional)
|-- LICENSE                    # MIT License
|-- README.md
|-- requirements.txt
|-- setup.py
|-- CITATION.cff
|-- .zenodo.json
```

## Model Architecture

### HydroAtmosFusion

| Component | Architecture | Input | Output |
|-----------|-------------|-------|--------|
| Ocean Encoder | ResNet-18 + SE Attention | (B, 3, 64, 64) | (B, 256, 16, 16) |
| Atmosphere Encoder | Swin-Transformer + ConvGRU | (B, 4, 3, 64, 64) | (B, 256, 16, 16) |
| Cross-Modal Attention | 8-head bidirectional | (B, 256, 16, 16) x2 | (B, 512, 16, 16) |
| Prediction Head | Conv1x1-BN-ReLU-Conv3x3-Sigmoid | (B, 512, 16, 16) | (B, 1, 64, 64) |

### Loss Function

L = 0.4 * L_WMSE + 0.35 * L_FocalCSI + 0.25 * L_FSS

## Results (Synthetic Data)

| Model | RMSE (mm/h) | CSI99 | EDI99 | FSS99 |
|-------|-------------|-------|-------|-------|
| ConvLSTM | 14.23 | 0.18 | 0.31 | 0.42 |
| CNN-LSTM | 12.56 | 0.22 | 0.38 | 0.51 |
| ViT-Precip | 11.82 | 0.25 | 0.42 | 0.55 |
| Pangu-Precip | 10.12 | 0.31 | 0.53 | 0.64 |
| **HydroAtmosFusion** | **8.47** | **0.38** | **0.62** | **0.74** |

Ocean state information contributes **37%** of extreme event prediction skill.

## Citation

If you use this code, please cite:

```bibtex
@article{panggabean2026hydroatmosfusion,
  title={HydroAtmosFusion (v1.0) and OceanReasonNet (v0.1): multimodal deep
         learning for extreme rainfall prediction using ocean--atmosphere coupling},
  author={Panggabean, Jogi and Purba, Noir Primadona},
  journal= On Proggress
  year={2026},
  publisher= -
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## Authors

- **Jogi Panggabean** - Torgas Laboratory, FPIK, Universitas Padjadjaran
- **Noir Primadona Purba** - Department of Marine Science, FPIK, Universitas Padjadjaran
