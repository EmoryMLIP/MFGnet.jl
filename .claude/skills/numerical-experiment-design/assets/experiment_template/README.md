# [Experiment Name]

## Overview

Brief description of what this experiment investigates.

**Research Question:** What specific question does this experiment answer?

**Hypothesis:** What do you expect to find?

## Setup

### Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data

- **Source:** Where the data comes from
- **Location:** `data/` directory
- **Preprocessing:** Description of any preprocessing steps

## Reproducibility

This experiment is configured for full reproducibility:

- **Random Seed:** Set in `configs/base_config.py`
- **PyTorch Version:** See `requirements.txt`
- **Hardware:** Document GPU/CPU used for main results
- **Deterministic Operations:** Enabled via `torch.backends.cudnn.deterministic = True`

## Running Experiments

### Single Run

```bash
python scripts/train.py --config configs/base_config.py
```

### Hyperparameter Sweep

```bash
python scripts/run_sweep.py --sweep configs/sweep_config.py
```

## Results

Results are saved to `results/[experiment_name]/[timestamp]/`:
- `metrics.json` - Training and validation metrics
- `model.pt` - Saved model checkpoint
- `config.json` - Experiment configuration used
- `environment.txt` - Python environment snapshot

## Analysis

See `notebooks/analysis.ipynb` for result visualization and statistical analysis.

## Project Structure

```
.
├── configs/           # Experiment configurations
├── data/             # Datasets (gitignored)
├── results/          # Experiment outputs (gitignored)
├── src/              # Source code (models, utilities)
├── scripts/          # Executable scripts
├── notebooks/        # Analysis notebooks
├── logs/             # Training logs (gitignored)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Citation

If you use this code, please cite:

```
[Add citation information]
```
