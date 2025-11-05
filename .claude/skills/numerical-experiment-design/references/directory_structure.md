# Directory Structure for Numerical Experiments

This document outlines best practices for organizing numerical experiment directories to ensure reproducibility, clarity, and ease of collaboration.

## Recommended Structure

```
experiment-name/
├── README.md              # Project overview and instructions
├── requirements.txt       # Python dependencies with versions
├── .gitignore            # Git ignore patterns
├── configs/              # Experiment configurations
│   ├── base_config.py
│   ├── sweep_config.py
│   └── ablation_config.py
├── data/                 # Datasets (gitignored, documented in README)
│   ├── raw/             # Original, immutable data
│   ├── processed/       # Cleaned, transformed data
│   └── README.md        # Data documentation
├── src/                  # Source code (models, utilities)
│   ├── __init__.py
│   ├── models/          # Model definitions
│   ├── data/            # Data loading and preprocessing
│   └── utils.py         # Utility functions
├── scripts/              # Executable scripts
│   ├── train.py         # Main training script
│   ├── evaluate.py      # Evaluation script
│   └── run_sweep.py     # Hyperparameter sweep
├── notebooks/            # Jupyter notebooks for analysis
│   ├── exploratory/     # Data exploration
│   └── analysis.ipynb   # Results analysis
├── results/              # Experiment outputs (gitignored)
│   └── [experiment]/[timestamp]/
│       ├── config.json          # Config used for this run
│       ├── metrics.json         # Training metrics
│       ├── model.pt             # Saved model
│       ├── environment.json     # Environment snapshot
│       └── tensorboard/         # TensorBoard logs
├── logs/                 # Training logs (gitignored)
├── tests/                # Unit and integration tests
│   ├── test_models.py
│   └── test_data.py
└── docs/                 # Additional documentation (optional)
    └── methodology.md
```

## Key Principles

### 1. Self-Containment

**The project should be fully self-contained**: someone should be able to clone the repository, install dependencies, and reproduce results without external dependencies (except documented datasets).

### 2. Clear Separation

Separate different types of content:
- **Code** (`src/`, `scripts/`) - Version controlled
- **Data** (`data/`) - Not version controlled, documented
- **Results** (`results/`, `logs/`) - Not version controlled, reproducible from code
- **Configuration** (`configs/`) - Version controlled
- **Analysis** (`notebooks/`) - Version controlled

### 3. Immutability of Raw Data

Raw data should never be modified:
- Keep original data in `data/raw/`
- Store processed data in `data/processed/`
- Document all preprocessing steps in code

### 4. Machine-Readable Naming

Use consistent, machine-readable naming conventions:

```
# Good
2023-01-15_experiment-name_lr-0.001_bs-32/

# Bad
experiment 1/
test (final)/
```

### 5. README is Critical

Every project must have a comprehensive README containing:
- What the project does
- How to set up the environment
- How to run experiments
- How to reproduce results
- Where to find things
- Expected computational requirements

## Directory Details

### configs/

Configuration files should be:
- Version controlled
- Human-readable (Python, YAML, JSON)
- Contain all hyperparameters
- Include experiment metadata

Example organization:
```
configs/
├── base_config.py          # Default configuration
├── sweep_config.py         # Hyperparameter sweep
├── ablation/               # Ablation studies
│   ├── no_dropout.py
│   └── smaller_model.py
└── production_config.py    # Best model configuration
```

### data/

Data organization:
```
data/
├── README.md              # Data documentation
├── raw/                   # Original data (never modified)
│   └── dataset_v1.csv
├── processed/             # Preprocessed data
│   └── dataset_clean.pt
└── download_data.sh       # Script to download data
```

**Important**:
- Document data sources and preprocessing in `data/README.md`
- Include checksums for data verification
- Never commit large data files to git

### results/

Results organization by experiment and timestamp:
```
results/
└── resnet_training/
    ├── 20230115_143022/       # Timestamp: YYYYMMDD_HHMMSS
    │   ├── config.json        # Exact config used
    │   ├── metrics.json       # All metrics
    │   ├── model.pt           # Saved model
    │   ├── environment.json   # Environment info
    │   ├── logs.txt          # Text logs
    │   └── tensorboard/       # TensorBoard logs
    └── 20230116_091530/
        └── ...
```

Benefits:
- Easy to compare different runs
- Timestamped for chronological tracking
- Self-documenting with saved configs
- No accidental overwrites

### scripts/

Executable scripts should:
- Be runnable from command line
- Accept arguments for configuration
- Have clear usage documentation
- Follow consistent naming

Example:
```
scripts/
├── train.py              # Main training
├── evaluate.py           # Evaluation
├── run_sweep.py          # Hyperparameter search
├── visualize.py          # Generate plots
└── preprocess_data.py    # Data preprocessing
```

### src/

Source code organization:
```
src/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── resnet.py
│   └── transformer.py
├── data/
│   ├── __init__.py
│   ├── datasets.py
│   └── transforms.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── losses.py
└── utils.py
```

### notebooks/

Organize notebooks by purpose:
```
notebooks/
├── exploratory/              # Exploratory analysis
│   ├── 01_data_exploration.ipynb
│   └── 02_model_prototyping.ipynb
├── analysis.ipynb           # Main results analysis
└── visualization.ipynb      # Figure generation
```

**Best practices**:
- Number notebooks in execution order
- Clear, descriptive names
- Clean outputs before committing (use nbstripout)
- Convert key notebooks to scripts for reproducibility

## Git Integration

### What to Commit

**Do commit**:
- Source code (`src/`, `scripts/`)
- Configurations (`configs/`)
- Documentation (README, docs)
- Notebooks (with clean outputs)
- Requirements file
- Test code

**Don't commit**:
- Data files (document how to obtain them)
- Results and logs
- Model checkpoints (too large, reproducible)
- Virtual environments
- IDE-specific files
- Temporary files

### .gitignore Template

```gitignore
# Data
data/
*.csv
*.hdf5
*.npy

# Results
results/
logs/
checkpoints/
*.pt
*.pth

# Python
__pycache__/
*.pyc
venv/

# Jupyter
.ipynb_checkpoints

# IDE
.vscode/
.idea/
```

## Documentation Requirements

### README.md

Must include:

```markdown
# Experiment Name

## Overview
Brief description of the experiment

## Setup
How to install dependencies and prepare the environment

## Data
Where to get the data, how to prepare it

## Running Experiments
Exact commands to reproduce results

## Results
Summary of key findings

## Citation
How to cite this work
```

### data/README.md

Document:
- Data source and how to obtain it
- Data format and structure
- Preprocessing steps
- Any data cleaning performed
- Train/val/test splits
- Data statistics

### Environment Documentation

Always save environment information with results:

```python
{
  "python_version": "3.9.7",
  "pytorch_version": "2.0.1",
  "cuda_version": "11.7",
  "gpu": "NVIDIA A100",
  "timestamp": "2023-01-15T14:30:22"
}
```

## Computational Efficiency

### Avoid Redundancy

- Don't duplicate code across scripts
- Use `src/` for shared functionality
- Import from `src/` in scripts and notebooks

### Modular Design

- Separate data loading, model definition, training logic
- Easy to swap components for experiments
- Facilitates testing

## Best Practices Summary

1. **Self-contained projects**: Everything needed to reproduce results
2. **Clear structure**: Consistent, predictable organization
3. **Comprehensive documentation**: README, code comments, docstrings
4. **Version control**: Track code, configs, and documentation
5. **Timestamped results**: Never overwrite previous experiments
6. **Environment tracking**: Save full environment with each experiment
7. **Machine-readable naming**: Consistent, parseable file/directory names
8. **Immutable raw data**: Never modify original data
9. **Automated workflows**: Scripts for common tasks
10. **Testing**: Unit tests for critical functionality

## References

- ENCORE Framework: https://www.nature.com/articles/s41467-024-52446-8
- Reproducible Data Science: https://ecorepsci.github.io/reproducible-science/
- Project Organization Best Practices: https://earthdatascience.org/courses/intro-to-earth-data-science/
