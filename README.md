# KAN-QSPR: Kolmogorov-Arnold Networks for Molecular Property Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![RDKit](https://img.shields.io/badge/RDKit-2026.03-blue.svg)](https://www.rdkit.org/)

**Symbolic QSPR modeling** using Kolmogorov-Arnold Networks (KAN) for automatic feature selection, model selection, and interpretable symbolic formula extraction across MoleculeNet datasets and kinase targets.

> 📄 This repository accompanies the paper *"KAN-based symbolic QSPR modeling for molecular property prediction"* (under review at *Computational Biology and Chemistry*).

## 🔬 Overview

This framework leverages the **Kolmogorov-Arnold theorem** to build inherently interpretable QSAR/QSPR models. Unlike traditional black-box deep learning approaches, KAN learns **symbolic mathematical expressions** that directly relate molecular descriptors to properties — providing both predictive power and mechanistic insight.

### Two-Stage Pipeline

```
Molecular Descriptors
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: Feature Selection │
│  • L1-regularized KAN       │
│  • Pareto-optimal pruning   │
│  • Automatic descriptor      │
│    importance ranking        │
└─────────────┬───────────────┘
              │ Selected features
              ▼
┌─────────────────────────────┐
│  Stage 2: Model Selection   │
│  • Grid search over (G, ε)  │
│  • Training + auto_symbolic │
│  • Pareto-optimal symbolic  │
│    formula extraction        │
└─────────────┬───────────────┘
              │
              ▼
    f(x) = 0.42·tanh(1.3x₁−0.7) + 0.18·sin(0.9x₂+1.1) + ...
```

1. **Feature Selection** — Train a narrow KAN with L1 regularization; prune inputs below a threshold; retain the Pareto-optimal descriptor subset that maximizes performance while minimizing feature count.
2. **Model Selection** — Grid search over KAN hyperparameters `(grid_size, grid_eps)`; train on selected features; apply `auto_symbolic()` to extract closed-form symbolic expressions; select the Pareto-optimal model balancing accuracy and formula simplicity.

## 📊 Datasets

All datasets use a uniform **80/10/10** (train/validation/test) split. BBBP and BACE use `ScaffoldSplitter` for chemically meaningful splits; all others use `RandomSplitter`.

| Type | Dataset | Metric | Samples | Descriptors |
|------|---------|--------|---------|-------------|
| Single-label Classification | **ClinTox** | ROC-AUC / F1 | 1,478 | Atom features + 5 RDKit |
| | **BACE** | ROC-AUC / F1 | 1,513 | Atom features + 5 RDKit |
| Multi-label Classification | **SIDER** | ROC-AUC / F1 | 1,427 | Atom features + 5 RDKit |
| | **HOB** | ROC-AUC / F1 | 8,430 | 14 custom descriptors |
| Regression | **ESOL** | RMSE / R² | 1,128 | Atom features + 16 custom |
| | **FreeSolv** | RMSE / R² | 642 | Atom features + 14 custom |
| | **CDK9** | RMSE / R² | 1,526 | 63 RDKit descriptors |
| | **BBBP** | ROC-AUC / F1 | 2,039 | Atom features + 5 RDKit |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU fallback supported)

### Installation

```bash
# Clone the repository
git clone https://github.com/lly418/KAN_QSPR.git
cd KAN_QSPR

# Create and activate conda environment (recommended)
conda create -n kan-qspr python=3.10 -y
conda activate kan-qspr

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ **Important**: This project uses a **modified version of pykan v0.2.7**. The modified files are in `pykan/` (`MultKAN.py`, `utils.py`). After `pip install pykan==0.2.7`, replace the installed files with the local `pykan/` versions:
> ```bash
> cp pykan/MultKAN.py $(python -c "import kan; print(kan.__path__[0])")/
> cp pykan/utils.py $(python -c "import kan; print(kan.__path__[0])")/
> ```

### Pre-computed Molecular Graphs

The `.bin` files (`bace_dglgraph.bin`, `bbbp_dglgraph.bin`, etc.) are pre-computed DGL molecular graphs. They are auto-generated on first run if absent, but pre-computing them saves significant time.

## 🧪 Running Experiments

### Main Entry Point

```bash
# Run all 8 datasets (single seed with grid search)
python reproduce_results.py

# Run a single dataset
python reproduce_results.py --dataset ESOL

# Multi-seed mode (uses pre-computed optimal hyperparameters)
python reproduce_results.py --seeds 42,456,23,123,789

# Force recompute (ignore cache)
python reproduce_results.py --no-cache

# Combine flags
python reproduce_results.py --dataset BACE --seeds 42,123,456 --no-cache
```

**Outputs** (saved to `results/`):
- `multi_seed_summary.csv` — mean ± std across seeds
- `multi_seed_per_seed.json` — per-seed raw metrics
- `*_featsdf.pkl` / `*_modelsdf.pkl` — cached grid search results

### Baseline Comparisons

```bash
# Traditional ML baselines (RF, XGBoost, MLP) on KAN-selected descriptors
python baselines/run_baselines.py                          # all 8 datasets, 5 seeds
python baselines/run_baselines.py --dataset BACE
python baselines/run_baselines.py --models rf,xgboost

# GNN baselines (GCN, GIN, GINE, SchNet, D-MPNN) on HOB & CDK9
python gnn_baselines/run.py                                # all datasets + models
python gnn_baselines/run.py --dataset HOB --models gcn,gin
```

### Statistical Analysis

```bash
# Paired t-tests + Wilcoxon signed-rank (KAN vs RF/XGBoost/MLP)
python revision/scripts/compute_E3_significance.py

# 95% confidence intervals
python revision/scripts/compute_E4_CI.py

# Symbolic formula stability across seeds
python revision/scripts/compute_F1_symbolic_sensitivity.py
```

## 🧠 Symbolic Formula Extraction

A key advantage of KAN is extracting **human-readable symbolic formulas** from trained models. After training, `auto_symbolic()` converts each learned activation function into a symbolic expression (e.g., `tanh`, `sin`, `exp`, `x²`, etc.), producing closed-form QSPR equations that relate molecular descriptors to target properties.

See `KAN_Formula.py` and `symbolic_expression/` for extracted formulas. Full results will be available in the accompanying paper.

## 📁 Project Structure

```
KAN_QSPR/
├── reproduce_results.py     # ★ Central experiment runner
├── utils/                   # Core pipeline modules
│   ├── data.py              # Data loading & DGL graph construction
│   ├── feature_selection.py # L1-regularized feature pruning
│   ├── model_selection.py   # Grid search + symbolic extraction
│   ├── molecule.py          # Molecular featurization (atom/bond)
│   ├── plotting.py          # Heatmaps, Pareto fronts, confusion matrices
│   └── meter.py             # Multi-label ROC-AUC metrics
├── pykan/                   # Modified pykan v0.2.7 (do not replace)
├── baselines/               # RF, XGBoost, MLP baselines
├── gnn_baselines/           # GCN, GIN, GINE, SchNet, D-MPNN baselines
├── experiments/             # Original per-dataset scripts (reference)
├── revision/                # Statistical tests & computed tables
├── data/                    # CSV datasets & feature descriptions
├── draw/                    # Paper figure scripts & PDF outputs
├── symbolic_expression/     # Extracted symbolic formulas
├── models/                  # GNN model architectures
├── KAN_Formula.py           # Catalog of extracted formulas
└── results/                 # Experiment outputs (CSV, pickle, PDF)
```

## 🔧 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.4.1 | Deep learning backend |
| pykan | 0.2.7 (modified) | KAN model implementation |
| DGL + DGLLife | 1.1.2 / 0.3.2 | Molecular graph construction |
| RDKit | 2026.03 | Molecular descriptor computation |
| XGBoost | 3.2.0 | Baseline model |
| SymPy | 1.13.3 | Symbolic formula manipulation |
| paretoset | 1.2.4 | Pareto-optimal solution selection |

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{kan_qspr_2025,
  title     = {KAN-based Symbolic QSPR Modeling for Molecular Property Prediction},
  author    = {Liu, Laiyu and {[co-authors]}},
  journal   = {Computational Biology and Chemistry},
  year      = {2025},
  note      = {Under review},
  url       = {https://github.com/lly418/KAN_QSPR}
}
```


## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ for interpretable AI in drug discovery</sub>
</p>
