#!/usr/bin/env python3
"""
Initialize a new numerical experiment directory structure.

This script creates a complete experiment directory with all necessary
subdirectories, template files, and best-practice configurations.

Usage:
    python init_experiment.py <experiment-name> [--path <output-directory>]

Example:
    python init_experiment.py resnet-training --path ~/experiments
"""

import argparse
import shutil
from pathlib import Path


EXPERIMENT_STRUCTURE = {
    'configs': 'Experiment configurations',
    'data': 'Datasets (gitignored)',
    'data/raw': 'Original, immutable data',
    'data/processed': 'Preprocessed data',
    'results': 'Experiment outputs (gitignored)',
    'src': 'Source code',
    'src/models': 'Model definitions',
    'src/data': 'Data loading utilities',
    'scripts': 'Executable scripts',
    'notebooks': 'Analysis notebooks',
    'notebooks/exploratory': 'Exploratory data analysis',
    'logs': 'Training logs (gitignored)',
    'tests': 'Unit tests'
}


def create_directory_structure(base_path: Path):
    """Create the experiment directory structure."""
    for directory in EXPERIMENT_STRUCTURE.keys():
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")


def create_readme(base_path: Path, experiment_name: str):
    """Create README.md from template."""
    readme_content = f"""# {experiment_name}

## Overview

Brief description of what this experiment investigates.

**Research Question:** What specific question does this experiment answer?

**Hypothesis:** What do you expect to find?

## Setup

### Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

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
├── tests/            # Unit tests
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Citation

If you use this code, please cite:

```
[Add citation information]
```
"""
    with open(base_path / 'README.md', 'w') as f:
        f.write(readme_content)
    print("✅ Created: README.md")


def create_requirements(base_path: Path):
    """Create requirements.txt."""
    requirements_content = """# Core Dependencies
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Experiment Tracking & Logging
tensorboard>=2.12.0
# wandb>=0.15.0  # Uncomment if using Weights & Biases
# mlflow>=2.5.0  # Uncomment if using MLflow

# Configuration Management
pyyaml>=6.0

# Plotting and Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
pandas>=2.0.0

# Development
pytest>=7.3.0
"""
    with open(base_path / 'requirements.txt', 'w') as f:
        f.write(requirements_content)
    print("✅ Created: requirements.txt")


def create_gitignore(base_path: Path):
    """Create .gitignore."""
    gitignore_content = """# Data directories
data/
datasets/
*.hdf5
*.h5
*.mat
*.npy
*.npz

# Results and outputs
results/
outputs/
logs/
runs/
checkpoints/
*.pt
*.pth
*.ckpt

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Experiment tracking
wandb/
mlruns/

# Temporary files
*.tmp
*.bak
*.log
"""
    with open(base_path / '.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✅ Created: .gitignore")


def create_init_files(base_path: Path):
    """Create __init__.py files for Python packages."""
    init_content = '"""Package initialization."""\n'

    for pkg in ['src', 'src/models', 'src/data', 'tests']:
        init_path = base_path / pkg / '__init__.py'
        with open(init_path, 'w') as f:
            f.write(init_content)

    print("✅ Created: __init__.py files")


def create_data_readme(base_path: Path):
    """Create data/README.md."""
    data_readme = """# Data Documentation

## Data Sources

Document where the data comes from and how to obtain it.

## Data Structure

Describe the format and structure of the data files.

## Preprocessing

Document all preprocessing steps applied to the data.

## Splits

Document train/validation/test splits:
- Training: X samples
- Validation: Y samples
- Test: Z samples

## Statistics

Include relevant data statistics (mean, std, class distributions, etc.)
"""
    with open(base_path / 'data' / 'README.md', 'w') as f:
        f.write(data_readme)
    print("✅ Created: data/README.md")


def main():
    parser = argparse.ArgumentParser(
        description='Initialize a new numerical experiment directory'
    )
    parser.add_argument(
        'experiment_name',
        type=str,
        help='Name of the experiment (will be used as directory name)'
    )
    parser.add_argument(
        '--path',
        type=str,
        default='.',
        help='Parent directory where experiment will be created (default: current directory)'
    )
    args = parser.parse_args()

    # Create base path
    base_path = Path(args.path) / args.experiment_name
    if base_path.exists():
        print(f"❌ Error: Directory {base_path} already exists!")
        return

    print(f"🚀 Initializing experiment: {args.experiment_name}")
    print(f"   Location: {base_path.absolute()}\n")

    # Create structure
    create_directory_structure(base_path)
    create_readme(base_path, args.experiment_name)
    create_requirements(base_path)
    create_gitignore(base_path)
    create_init_files(base_path)
    create_data_readme(base_path)

    print(f"\n✅ Experiment '{args.experiment_name}' initialized successfully!")
    print(f"\nNext steps:")
    print(f"1. cd {args.experiment_name}")
    print(f"2. python -m venv venv && source venv/bin/activate")
    print(f"3. pip install -r requirements.txt")
    print(f"4. Update configs/base_config.py with your hyperparameters")
    print(f"5. Implement your model in src/models/")
    print(f"6. Create your training script in scripts/train.py")


if __name__ == '__main__':
    main()
