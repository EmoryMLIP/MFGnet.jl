---
name: numerical-experiment-design
description: This skill should be used when setting up directory structures, scripts, and configurations for reproducible numerical experiments in PyTorch. It provides templates, best practices, and utilities for experiment management, hyperparameter tracking, and ensuring scientific validity. Use this skill when creating new experiment projects, ensuring reproducibility, managing hyperparameters, or setting up post-processing workflows for numerical computing research.
---

# Numerical Experiment Design

## Overview

This skill provides comprehensive support for designing and implementing reproducible numerical experiments using PyTorch. It includes directory structure templates, reproducibility utilities, hyperparameter management strategies, and best practices for scientific computing. The skill ensures experiments follow established reproducibility standards and can be easily shared, understood, and reproduced by others.

## When to Use This Skill

Invoke this skill when:
- Setting up a new numerical experiment or research project
- Ensuring reproducibility of existing PyTorch code
- Implementing hyperparameter sweeps or ablation studies
- Creating directory structures for machine learning experiments
- Setting up post-processing and analysis workflows
- Preparing experiments for publication or sharing

## Core Capabilities

### 1. Complete Experiment Initialization

Create fully-structured experiment directories with all necessary components using the provided initialization script.

**Using the initialization script:**

```bash
python scripts/init_experiment.py <experiment-name> --path <output-directory>
```

This creates a complete experiment structure with:
- Organized directory hierarchy
- Configuration management system
- README template with best practices
- .gitignore configured for experiment data
- Python package structure with __init__.py files
- Data documentation templates

**Alternative: Copy template directly**

Copy the complete experiment template from `assets/experiment_template/` and customize as needed. This template includes all files and structure needed for a reproducible experiment.

### 2. Reproducibility Configuration

Ensure experiments produce consistent, reproducible results across runs.

**Core reproducibility function:**

Use the reproducibility utilities from `scripts/reproducibility_utils.py`:

```python
from reproducibility_utils import set_seed, save_environment

# Set all random seeds for reproducibility
set_seed(42, deterministic=True)

# Save environment information with results
save_environment(output_dir / "environment.json")
```

**Key reproducibility requirements (PyTorch-specific):**

1. **Comprehensive seed setting:**
   ```python
   import random
   import numpy as np
   import torch

   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   ```

2. **CuDNN deterministic operations:**
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   torch.use_deterministic_algorithms(True, warn_only=True)
   ```

3. **DataLoader worker seeding:**
   ```python
   from torch.utils.data import DataLoader

   train_loader = DataLoader(
       dataset,
       batch_size=32,
       num_workers=4,
       worker_init_fn=seed_worker,  # From reproducibility_utils
       generator=torch.Generator().manual_seed(42),
       drop_last=True  # Ensures consistent batch sizes
   )
   ```

**Performance vs. reproducibility trade-off:**
- Deterministic mode (`deterministic=True`) ensures reproducibility but may be 20-50% slower
- Document this choice in your experiment configuration
- For final publication results, use deterministic mode

**For detailed PyTorch reproducibility information:**
- Read `references/pytorch_reproducibility.md` for comprehensive best practices
- Includes debugging tips, caveats, and platform-specific considerations

### 3. Hyperparameter Management

Implement structured configuration systems for managing hyperparameters.

**Configuration approach (Python class-based):**

The experiment template includes `configs/base_config.py` with a complete configuration class:

```python
from configs.base_config import Config

config = Config()
config.model.hidden_dim = 256  # Easy to modify
config.training.learning_rate = 1e-3

# Save with results
with open(output_dir / 'config.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2)
```

**Hyperparameter sweeps:**

For hyperparameter optimization, use the sweep configuration template in `configs/sweep_config.py`:

```python
sweep_params = {
    "training.learning_rate": [1e-4, 5e-4, 1e-3],
    "model.hidden_dim": [64, 128, 256],
    "training.weight_decay": [1e-6, 1e-5, 1e-4],
}
```

**Recommended tools for hyperparameter optimization:**
- Grid search: Small parameter spaces, understanding interactions
- Random search: Large spaces, limited compute budget
- Optuna (Bayesian): Expensive evaluations, want to minimize trials
- Weights & Biases: Cloud-based sweeps with visualization

**For comprehensive hyperparameter management guidance:**
- Read `references/hyperparameter_management.md` for detailed strategies
- Includes configuration systems, sweep methods, logging, and best practices

### 4. Directory Structure Best Practices

Organize experiments following scientific computing reproducibility standards.

**Recommended structure (from experiment template):**

```
experiment-name/
├── configs/          # Experiment configurations (version controlled)
├── data/            # Datasets (gitignored, documented)
│   ├── raw/        # Original, immutable data
│   └── processed/  # Preprocessed data
├── src/             # Source code (version controlled)
│   ├── models/     # Model definitions
│   └── data/       # Data loading utilities
├── scripts/         # Executable scripts (version controlled)
├── results/         # Experiment outputs (gitignored, timestamped)
│   └── [experiment]/[timestamp]/
│       ├── config.json
│       ├── metrics.json
│       ├── model.pt
│       └── environment.json
├── notebooks/       # Analysis notebooks (version controlled)
├── logs/           # Training logs (gitignored)
└── README.md       # Documentation (version controlled)
```

**Key principles:**
- **Self-containment**: Complete project can be shared and reproduced
- **Clear separation**: Code, data, results, and configs are separated
- **Immutability**: Raw data never modified
- **Timestamped results**: Never overwrite previous experiments
- **Machine-readable naming**: Consistent, parseable file/directory names

**For detailed directory organization guidance:**
- Read `references/directory_structure.md` for comprehensive best practices
- Includes git integration, documentation requirements, and efficiency tips

### 5. Experiment Tracking and Logging

Implement systematic logging to track experiments and facilitate comparison.

**Minimal logging requirements:**

Every experiment run should save:
1. **Configuration** (`config.json`) - Exact hyperparameters used
2. **Environment** (`environment.json`) - PyTorch, CUDA, hardware info
3. **Metrics** (`metrics.json`) - Training and validation metrics over time
4. **Model** (`model.pt`) - Trained model checkpoint
5. **Logs** (`logs.txt`) - Text logs of training process

**Timestamped results:**

Organize results by timestamp to prevent overwrites:

```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = results_dir / experiment_name / timestamp
exp_dir.mkdir(parents=True, exist_ok=True)
```

**Experiment tracking tools:**

- **TensorBoard** (built-in): Basic metric logging and visualization
- **Weights & Biases**: Cloud-based tracking with hyperparameter sweeps
- **MLflow**: Open-source experiment tracking and model registry

The experiment template includes TensorBoard integration in `scripts/train.py`.

### 6. Template Files and Examples

The skill includes complete, ready-to-use template files in `assets/experiment_template/`:

**Configuration files:**
- `configs/base_config.py` - Main experiment configuration
- `configs/sweep_config.py` - Hyperparameter sweep configuration

**Source code:**
- `src/utils.py` - Common utilities (seed setting, checkpointing, etc.)
- `src/__init__.py` - Package initialization

**Scripts:**
- `scripts/train.py` - Complete training script with reproducibility measures

**Documentation:**
- `README.md` - Comprehensive project README template
- `data/README.md` - Data documentation template

**Infrastructure:**
- `.gitignore` - Configured for numerical experiments
- `requirements.txt` - Common dependencies with versions

## Workflow Guide

### Workflow 1: Creating a New Experiment

1. **Initialize the experiment structure:**
   ```bash
   python scripts/init_experiment.py my-experiment --path ~/experiments
   cd ~/experiments/my-experiment
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure experiment:**
   - Edit `configs/base_config.py` with your hyperparameters
   - Update README.md with experiment description
   - Document data sources in `data/README.md`

4. **Implement model and data loading:**
   - Create model in `src/models/`
   - Implement data loading in `src/data/`
   - Add utilities to `src/utils.py`

5. **Create training script:**
   - Use `scripts/train.py` as template
   - Ensure reproducibility measures are implemented
   - Add proper logging and checkpointing

6. **Run experiment:**
   ```bash
   python scripts/train.py --config configs/base_config.py
   ```

7. **Analyze results:**
   - Results automatically saved to `results/[experiment]/[timestamp]/`
   - Create analysis notebook in `notebooks/`

### Workflow 2: Ensuring Reproducibility of Existing Code

1. **Add seed setting at the beginning:**
   ```python
   from reproducibility_utils import set_seed
   set_seed(42, deterministic=True)
   ```

2. **Configure CuDNN for determinism:**
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

3. **Fix DataLoader:**
   ```python
   from reproducibility_utils import seed_worker

   loader = DataLoader(
       dataset,
       worker_init_fn=seed_worker,
       generator=torch.Generator().manual_seed(42),
       drop_last=True
   )
   ```

4. **Save environment and config:**
   ```python
   from reproducibility_utils import save_environment
   save_environment(output_dir / "environment.json")

   with open(output_dir / "config.json", "w") as f:
       json.dump(config, f, indent=2)
   ```

5. **Verify reproducibility:**
   ```python
   from reproducibility_utils import verify_reproducibility
   verify_reproducibility(train_function, seed=42, n_runs=2)
   ```

### Workflow 3: Hyperparameter Sweep

1. **Create sweep configuration:**
   - Copy and modify `configs/sweep_config.py`
   - Define parameter grid or ranges

2. **Choose sweep method:**
   - Grid search for small spaces
   - Random search for exploration
   - Bayesian (Optuna) for expensive evaluations

3. **Implement sweep script:**
   ```python
   for params in param_combinations:
       config = update_config(base_config, params)
       results = run_experiment(config)
       log_results(results)
   ```

4. **Analyze sweep results:**
   - Compare metrics across configurations
   - Identify best hyperparameters
   - Document findings

Refer to `references/hyperparameter_management.md` for detailed implementation examples.

## Best Practices Summary

1. **Always set random seeds** before any random operations
2. **Save exact configuration** with every experiment run
3. **Document environment** (PyTorch version, CUDA, hardware)
4. **Use timestamped directories** to avoid overwriting results
5. **Pin dependency versions** in requirements.txt
6. **Keep raw data immutable** - preprocessing creates new files
7. **Version control code and configs** - not data or results
8. **Test reproducibility** with multiple runs using same seed
9. **Document hardware** used for main results
10. **Include comprehensive README** for every experiment

## Caveats and Limitations

**Cross-platform reproducibility:**
- Results NOT guaranteed across different PyTorch versions
- Results NOT guaranteed across CPU vs. GPU
- Results NOT guaranteed across different GPU models
- Results NOT guaranteed across different operating systems

**What IS reproducible:**
- Same hardware + same software versions + proper seeding

**Performance impact:**
- Deterministic mode can be 20-50% slower than default
- Trade-off between reproducibility and speed
- Document this choice in experiment configuration

**Operations without deterministic implementations:**
- Some PyTorch operations don't have deterministic alternatives
- Use `warn_only=True` to identify these operations
- See `references/pytorch_reproducibility.md` for details

## Resources

### scripts/

**`init_experiment.py`**
Executable script to create new experiment directory structures. Run directly to initialize a complete experiment project with all necessary files and folders.

**`reproducibility_utils.py`**
Python module with reproducibility functions. Can be imported in your code or run standalone to check environment. Provides:
- `set_seed()` - Comprehensive seed setting
- `save_environment()` - Environment documentation
- `seed_worker()` - DataLoader worker seeding
- `verify_reproducibility()` - Test if code is reproducible
- `create_reproducibility_report()` - Generate full reproducibility documentation

### references/

**`pytorch_reproducibility.md`**
Comprehensive guide to PyTorch reproducibility. Read this for:
- Detailed explanation of all random sources in PyTorch
- CuDNN configuration options
- DataLoader reproducibility patterns
- Debugging non-reproducible code
- Platform-specific caveats
- Complete reproducibility checklist

**`directory_structure.md`**
Best practices for organizing numerical experiments. Read this for:
- Detailed directory structure recommendations
- Git integration strategies
- Documentation requirements
- Self-contained project principles
- Naming conventions
- ENCORE framework reference

**`hyperparameter_management.md`**
Strategies for managing hyperparameters. Read this for:
- Configuration system comparisons (Python classes, YAML, Hydra)
- Hyperparameter sweep strategies (grid, random, Bayesian)
- Experiment tracking tools (TensorBoard, W&B, MLflow)
- Complete sweep implementation examples
- Configuration validation patterns
- Best practices for hyperparameter naming and documentation

### assets/

**`experiment_template/`**
Complete experiment directory structure ready to copy and customize. Includes:
- Full directory hierarchy (`configs/`, `data/`, `src/`, `scripts/`, `notebooks/`, `results/`, `logs/`)
- Configuration templates (`base_config.py`, `sweep_config.py`)
- Training script template (`scripts/train.py`)
- Source code utilities (`src/utils.py`)
- Documentation templates (`README.md`, `data/README.md`)
- Infrastructure files (`.gitignore`, `requirements.txt`)

Use this when you want to manually copy the structure instead of using the initialization script.

## Additional Notes

**When to read reference files:**
- Read `references/pytorch_reproducibility.md` when debugging reproducibility issues or when setting up a new experiment for the first time
- Read `references/directory_structure.md` when designing custom directory organizations or preparing for publication
- Read `references/hyperparameter_management.md` when implementing hyperparameter sweeps or using advanced configuration systems

**Scripts can be imported:**
Both scripts in `scripts/` are designed to be either executed standalone or imported as Python modules:
```python
# Import and use
from reproducibility_utils import set_seed, save_environment
set_seed(42)
```

**Template customization:**
The experiment template is intentionally comprehensive. Delete any components you don't need:
- Remove `notebooks/` if not doing exploratory analysis
- Remove sweep config if only running single experiments
- Simplify directory structure for smaller projects
- Adapt to your specific workflow
