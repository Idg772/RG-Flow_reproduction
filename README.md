# IGF23 — RG‑Flow
This is my Data Analysis project for the MPhil in Data Intensive Science at the University of Cambridge
The aim of this work is to investigate and reproduce the results of "RG-Flow: A hierarchical and explainable flow model based on renormalization group and sparse prior" ([arXiv:2010.00029](https://arxiv.org/abs/2010.00029)).
A PyTorch implementation of Renormalization Group Flow (RG-Flow) for generative modeling with hierarchical image synthesis.

---

## 1. Repository Structure

```
IGF23/
├── data/                    ← Datasets (auto-generated when needed)
├── notebooks/
│   ├── GoogleColab/         ← Scripts used on Colab for training
│   └── analysis.ipynb       ← Exploration notebook
├── report/
│   ├── report.pdf           ← Final Report
│   └── summary.pdf          ← Executive summary
├── runs/                    ← Training outputs (logs, checkpoints, samples)
├── src/
│   ├── data/                ← Dataset loaders and preprocessing
│   │   ├── celeba.py
│   │   └── msds/            ← Synthetic dataset generators
│   │       ├── generate_msds1.py
│   │       └── generate_msds2.py
│   ├── distributions/       ← Prior distributions (Laplace, Gaussian)
│   │   ├── __init__.py
│   │   └── distribution.py
│   ├── layers/              ← Model architecture components
│   │   ├── __init__.py
│   │   ├── build_network.py ← Main model builder
│   │   ├── Decimator.py     ← Spatial downsampling
│   │   ├── Disentangler.py  ← Channel manipulation
│   │   ├── MERA.py          ← Core RG-Flow implementation
│   │   ├── RNVP.py          ← Real NVP coupling layers
│   │   └── ResidualNetwork.py
│   ├── utils/               ← Training utilities
│   │   ├── __init__.py
│   │   ├── data_utils.py    ← Dataset loading helpers
│   │   ├── logging.py       ← Logging configuration
│   │   ├── sampling.py      ← Image generation utilities
│   │   └── transforms.py    ← Logit transforms
│   └── train.py             ← Main training script
├── tests/                   ← PyTest unit tests
│   ├── test_disentangler.py
│   └── test_rnvp.py
├── weights/                 ← Pre-trained model checkpoints
│   ├── gaussian.pth         ← Gaussian prior (32×32 CelebA)
│   └── laplacian.pth        ← Laplace prior (32×32 CelebA)
└── README.md                ← You are here
```

## 2. Quick Start

### 2.1 Install Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2.2 Quick Test Run

Test that everything works with a small subset of data:

```bash
python src/train.py --test-mode --epochs 1 --batch-size 32
```

This will:
- Use only 128 training samples for quick verification
- Train for 1 epoch (~30 seconds)
- Generate sample images
- Save checkpoints and logs to `runs/`

## 3. Training

### 3.1 Basic Usage

```bash
python src/train.py [OPTIONS]
```

### 3.2 Command Line Arguments

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--dataset` | `msds1`, `msds2`, `celeba` | `msds1` | Dataset to train on |
| `--data-root` | Path | `data` | Root folder for datasets |
| `--epochs` | Integer | `60` | Number of training epochs |
| `--batch-size` | Integer | `512` | Training batch size |
| `--lr` | Float | `1e-3` | Learning rate |
| `--prior` | `laplace`, `gaussian` | `laplace` | Prior distribution type |
| `--out` | Path | `runs` | Output directory for artifacts |
| `--device` | String | `auto` | Device to train on (`cpu` or `cuda`) |
| `--workers` | Integer | `8` | Number of data loading workers |
| `--img-samples` | Integer | `16` | Images sampled per epoch |
| `--test-mode` | Flag | `False` | Use small subset for quick testing |

### 3.3 Training Examples

**Standard training on MSDS1 with Laplace prior:**
```bash
python src/train.py --dataset msds1 --epochs 60 --batch-size 512
```

**Training on MSDS2 with Gaussian prior:**
```bash
python src/train.py --dataset msds2 --prior gaussian --epochs 100
```

**Quick test with different parameters:**
```bash
python src/train.py --test-mode --dataset msds2 --prior gaussian --img-samples 9
```

**CPU training:**
```bash
python src/train.py --test-mode --device cpu --batch-size 16
```

### 3.4 Output Structure

After training, you'll find outputs in the `runs/` directory:

```
runs/msds1_20250730_185240/
├── ckpt/
│   └── rgflow_ep001.pth     ← Model checkpoints
├── logs/
│   └── rgflow_*.log         ← Training logs
└── samples/
    └── epoch001.png         ← Generated sample images
```

## 4. Datasets

### 4.1 Supported Datasets

- **MSDS1**: Synthetic Multi-Scale Dsprite 1 dataset (auto-generated)
- **MSDS2**: Synthetic Multi-Scale Dsprite 2 dataset (auto-generated)  
- **CelebA**: Celebrity faces dataset (requires manual download)

### 4.2 Dataset Generation

MSDS datasets are automatically generated when first used:
- **Training set**: 90,000 samples
- **Test set**: 10,000 samples
- **Image size**: 32×32×3
- **Storage**: Saved to `data/msds1/` or `data/msds2/`

## 5. Model Architecture

### 5.1 RG-Flow Components

- **MERA Blocks**: Multi-scale Entanglement Renormalization Ansatz gates
- **RNVP Layers**: Real-valued Non-Volume Preserving transformations


### 5.2 Model Configuration

Default model parameters:
- **Depth**: 8 layers
- **Blocks per layer**: 4
- **Parameters**: 100,064,780 
- **Optimizer**: AdamW with weight decay 5e-5
- **Gradient clipping**: 1.0

## 6. Monitoring Training

### 6.1 Training Logs

Monitor training progress through:
- **Console output**: Real-time training metrics
- **Log files**: Detailed logs saved to `runs/*/logs/`
- **Sample images**: Generated every epoch in `runs/*/samples/`

### 6.2 Key Metrics

- **Loss**: Negative log-likelihood
- **BPD**: Bits per dimension (lower is better)
- **Validation BPD**: Generalization metric

### 6.3 Example Training Output

```
2025-07-30 18:52:40  INFO      Dataset msds1 (TEST MODE - small subset) – 4 training batches
2025-07-30 18:52:50  INFO      Image dimensions: C=3, H=32, W=32
2025-07-30 18:52:50  INFO      MERA model 80.52 M parameters; optimiser AdamW lr 1.0e-03
2025-07-30 18:53:13  INFO      [ep   1] batch    4/4  loss 167.1968  bpd 8.0785
2025-07-30 18:53:15  INFO      Epoch 1 finished in 25.1s  avg‑loss 451.8507
2025-07-30 18:53:27  INFO      Validation BPD: 8.5235
2025-07-30 18:53:27  INFO      Saved samples → runs/msds1_*/samples/epoch001.png
2025-07-30 18:53:28  INFO      Checkpoint saved → runs/msds1_*/ckpt/rgflow_ep001.pth
```

## 7. Development

### 7.1 Code Organization

The codebase follows modular design principles:

- **`src/train.py`**: Main training script and CLI
- **`src/utils/`**: Reusable training utilities
- **`src/layers/`**: Model architecture components
- **`src/data/`**: Dataset loading and preprocessing
- **`src/distributions/`**: Prior distribution implementations

### 7.2 Testing

Run unit tests:
```bash
pytest tests/
```

Quick functionality test:
```bash
python src/train.py --test-mode --epochs 1
```

### 7.3 Adding New Datasets

1. Create dataset loader in `src/data/`
2. Add import to `src/utils/data_utils.py`
3. Update `get_dataloaders()` function
4. Add dataset choice to `train.py` CLI

## 8. Pre-trained Models

Pre-trained checkpoints are available in `weights/`:
- **`gaussian.pth`**: Trained with Gaussian prior on CelebA
- **`laplacian.pth`**: Trained with Laplace prior on CelebA

## 9. AI Declaration

All code was created in VSCode with GitHub Copilot enabled. ChatGPT was used to help create and improve the quality of plots as well as to proofread the report. Some initial training was done on Google Colab with Gemini enabled.
