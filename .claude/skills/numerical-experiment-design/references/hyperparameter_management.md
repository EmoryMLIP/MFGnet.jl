# Hyperparameter Management for Numerical Experiments

This document covers strategies and best practices for managing hyperparameters in PyTorch-based numerical experiments, including configuration systems, sweep strategies, and tracking tools.

## Configuration Management Approaches

### 1. Python Class-Based Configs

**Pros**: Type hints, IDE support, inheritance, validation
**Cons**: Less readable for non-programmers

```python
from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)
```

Usage:
```python
config = Config()
config.model.hidden_dim = 256
```

### 2. YAML-Based Configs

**Pros**: Human-readable, language-agnostic, easy to edit
**Cons**: No type checking, runtime errors

```yaml
# config.yaml
model:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100

seed: 42
```

Loading:
```python
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

### 3. Hydra Framework (Recommended for Complex Projects)

**Pros**: Composition, overrides from CLI, multi-run, type-safe
**Cons**: Learning curve, additional dependency

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="base_config", version_base=None)
def main(cfg: DictConfig):
    print(cfg.model.hidden_dim)
    # Your training code here

if __name__ == "__main__":
    main()
```

CLI override:
```bash
python train.py model.hidden_dim=256 training.learning_rate=0.01
```

### 4. argparse + JSON

**Pros**: Simple, flexible, CLI-friendly
**Cons**: Verbose for many parameters

```python
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/base.json')
parser.add_argument('--lr', type=float, default=None)
args = parser.parse_args()

# Load base config
with open(args.config, 'r') as f:
    config = json.load(f)

# Override with CLI args
if args.lr is not None:
    config['training']['learning_rate'] = args.lr
```

## Hyperparameter Sweep Strategies

### 1. Grid Search

Exhaustive search over all combinations.

```python
import itertools

# Define parameter grid
param_grid = {
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'hidden_dim': [64, 128, 256],
    'dropout': [0.1, 0.2, 0.3]
}

# Generate all combinations
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Run experiments
for params in combinations:
    run_experiment(params)
```

**When to use**: Small number of parameters, understanding parameter interactions

**Computational cost**: Exponential in number of parameters

### 2. Random Search

Random sampling from parameter distributions.

```python
import numpy as np

def sample_params(n_trials=20):
    params = []
    for _ in range(n_trials):
        param = {
            'learning_rate': 10 ** np.random.uniform(-5, -2),
            'hidden_dim': np.random.choice([64, 128, 256, 512]),
            'dropout': np.random.uniform(0.0, 0.5)
        }
        params.append(param)
    return params

# Run experiments
for params in sample_params(n_trials=50):
    run_experiment(params)
```

**When to use**: Large parameter space, limited compute budget

**Advantage**: Often finds good configurations faster than grid search

### 3. Optuna (Bayesian Optimization)

Intelligent search using Bayesian optimization.

```python
import optuna

def objective(trial):
    # Define hyperparameter space
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)

    # Train model with these params
    val_loss = train_and_validate(lr, hidden_dim, dropout)

    return val_loss

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

**When to use**: Expensive evaluations, want to minimize trials

**Features**:
- Early stopping (pruning unpromising trials)
- Parallel optimization
- Visualization tools

### 4. Weights & Biases Sweeps

Cloud-based hyperparameter sweeps with visualization.

```python
import wandb

# Define sweep configuration
sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'hidden_dim': {
            'values': [64, 128, 256, 512]
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")

# Run sweep
def train():
    wandb.init()
    config = wandb.config
    # Train with wandb.config parameters
    # wandb.log({'val_loss': loss})

wandb.agent(sweep_id, train, count=50)
```

## Logging and Tracking

### 1. Save Configuration with Results

Always save the exact configuration used:

```python
import json
from pathlib import Path

def save_config(config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
```

### 2. Experiment Tracking Tools

#### TensorBoard (Built-in)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f'runs/{experiment_name}')

# Log hyperparameters and metrics
writer.add_hparams(
    {'lr': config.learning_rate, 'hidden_dim': config.hidden_dim},
    {'hparam/val_loss': val_loss, 'hparam/val_acc': val_acc}
)
```

#### Weights & Biases

```python
import wandb

wandb.init(project="my-project", config=config)

# Automatic logging
wandb.log({'train_loss': loss, 'val_loss': val_loss})

# Log model
wandb.save('model.pt')
```

#### MLflow

```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(config)

    # Log metrics
    mlflow.log_metric("val_loss", val_loss)

    # Log artifacts
    mlflow.log_artifact("model.pt")
```

### 3. Structured Logging

Create a logging utility:

```python
import json
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    def __init__(self, experiment_name, output_dir):
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / experiment_name / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = []

    def log_config(self, config):
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def log_metric(self, epoch, metrics):
        metrics['epoch'] = epoch
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics.append(metrics)

    def save_metrics(self):
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def get_output_dir(self):
        return self.output_dir
```

Usage:
```python
logger = ExperimentLogger("resnet_training", "results")
logger.log_config(config.to_dict())

for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = validate()
    logger.log_metric(epoch, {'train_loss': train_loss, 'val_loss': val_loss})

logger.save_metrics()
```

## Best Practices

### 1. Configuration Hierarchy

Use inheritance for related configurations:

```python
class BaseConfig:
    seed = 42
    batch_size = 32

class SmallModelConfig(BaseConfig):
    hidden_dim = 64
    num_layers = 2

class LargeModelConfig(BaseConfig):
    hidden_dim = 256
    num_layers = 6
```

### 2. CLI Overrides

Allow command-line overrides of config values:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='base')
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--batch_size', type=int, default=None)
args = parser.parse_args()

config = load_config(args.config)
if args.lr is not None:
    config.learning_rate = args.lr
if args.batch_size is not None:
    config.batch_size = args.batch_size
```

### 3. Validation

Validate configuration values:

```python
class Config:
    def __post_init__(self):
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 <= self.dropout < 1, "Dropout must be in [0, 1)"
        assert self.batch_size > 0, "Batch size must be positive"
```

### 4. Documentation

Document all hyperparameters:

```python
@dataclass
class Config:
    learning_rate: float = 1e-3
    """Learning rate for optimizer. Typical range: [1e-5, 1e-2]"""

    hidden_dim: int = 128
    """Hidden dimension for model layers. Must be > 0"""
```

### 5. Reproducibility

Always include seed in configuration:

```python
config = {
    'seed': 42,
    'deterministic': True,
    # ... other params
}

set_seed(config['seed'], config['deterministic'])
```

### 6. Naming Conventions

Use descriptive experiment names that encode key parameters:

```python
def create_experiment_name(config):
    return f"{config.model_name}_lr{config.learning_rate}_bs{config.batch_size}_wd{config.weight_decay}"

# Example: "resnet_lr0.001_bs32_wd1e-05"
```

### 7. Parameter Ranges

Document reasonable ranges for each parameter:

```python
HYPERPARAMETER_RANGES = {
    'learning_rate': {
        'min': 1e-5,
        'max': 1e-1,
        'scale': 'log',
        'default': 1e-3
    },
    'batch_size': {
        'min': 8,
        'max': 256,
        'scale': 'log2',  # Powers of 2
        'default': 32
    },
    'dropout': {
        'min': 0.0,
        'max': 0.5,
        'scale': 'linear',
        'default': 0.1
    }
}
```

## Hyperparameter Sweep Template

Complete example combining best practices:

```python
import itertools
import json
from pathlib import Path
from datetime import datetime

def run_sweep(base_config, param_grid, method='grid'):
    """
    Run hyperparameter sweep.

    Args:
        base_config: Base configuration dict
        param_grid: Dict of param_name -> [values]
        method: 'grid' or 'random'
    """
    # Create sweep directory
    sweep_name = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir = Path('results') / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Generate parameter combinations
    if method == 'grid':
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    elif method == 'random':
        # Implement random sampling
        combinations = sample_random_params(param_grid, n_trials=50)

    # Save sweep configuration
    sweep_config = {
        'base_config': base_config,
        'param_grid': param_grid,
        'method': method,
        'n_trials': len(combinations)
    }
    with open(sweep_dir / 'sweep_config.json', 'w') as f:
        json.dump(sweep_config, f, indent=2)

    # Run experiments
    results = []
    for i, params in enumerate(combinations):
        # Merge with base config
        config = {**base_config, **params}

        # Create experiment name
        exp_name = f"trial_{i:04d}"

        # Run experiment
        metrics = run_experiment(config, sweep_dir / exp_name)

        # Save results
        results.append({
            'trial_id': i,
            'config': config,
            'metrics': metrics
        })

    # Save all results
    with open(sweep_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Find best configuration
    best_trial = min(results, key=lambda x: x['metrics']['val_loss'])
    print(f"Best trial: {best_trial['trial_id']}")
    print(f"Best config: {best_trial['config']}")
    print(f"Best val_loss: {best_trial['metrics']['val_loss']}")

    return results
```

## Summary

1. **Use structured configs**: Python classes, YAML, or Hydra
2. **Version control configs**: Track configuration files in git
3. **Save configs with results**: Always save exact config used
4. **Enable CLI overrides**: Allow experimentation without editing files
5. **Validate configurations**: Catch errors early
6. **Document parameters**: Explain meaning and reasonable ranges
7. **Use appropriate search**: Grid for small spaces, Bayesian for expensive evaluations
8. **Track experiments**: TensorBoard, W&B, or MLflow
9. **Reproducible sweeps**: Set seeds, save environments
10. **Analyze systematically**: Compare results across hyperparameter values

## References

- Hydra: https://hydra.cc/
- Optuna: https://optuna.org/
- Weights & Biases: https://wandb.ai/
- MLflow: https://mlflow.org/
