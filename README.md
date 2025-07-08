# IGF23 — RG‑Flow

---

## 1. Repository Structure

```
IGF23/
├── data/                    ← (optional) where the downloaded datasets go
├── notebooks/
│   ├── GoogleColab/         ← Scripts used on Colab for training
│   └── analysis.ipynb       ← Exploration notebook
├── report/
│   ├── report.pdf           ← Final Report
│   └── summary.pdf          ← Executive summary
├── src/
│   ├── data/                ← Helpers to load / preprocess datasets
│   ├── msds/                ← Synthetic dataset generators
│   ├── distributions/       ← Base latent distributions
│   ├── layers/              ← All model code here
│   │   ├── build_network.py
│   │   ├── Decimator.py
│   │   ├── Disentangler.py
│   │   ├── MERA.py
│   │   ├── RNVP.py
│   │   ├── ResidualNetwork.py
│   │   └── train.py
│   └── tests/               ← PyTest unit tests for critical layers, used in development for verification
├── weights/
│   ├── gaussian.pth         ← Pre‑trained RG‑Flow checkpoint (32 × 32 CelebA)
│   └── laplacian.pth
└── README.md                ← You are here
```

**Note:** MSDS weights were not included due to size constraints.

## 2. Quick Start

### 2.1 Install Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. AI Declaration

All code was created in VSCode with GitHub Copilot enabled. ChatGPT was used to help create and improve the quality of plots as well as to proofread the report. Some initial training was done on Google Colab with Gemini enabled.