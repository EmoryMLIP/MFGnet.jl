"""
Hyperparameter sweep configuration.

Defines parameter ranges for grid search or random search.
"""

from configs.base_config import Config

class SweepConfig(Config):
    """Configuration for hyperparameter sweeps."""

    # Base configuration
    experiment_name = "hyperparameter_sweep"
    description = "Grid search over learning rates and hidden dimensions"

    # Sweep method: 'grid', 'random', 'optuna'
    sweep_method = "grid"

    # Number of trials for random search
    n_trials = 20

    # Parameters to sweep (grid search)
    sweep_params = {
        "training.learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
        "model.hidden_dim": [64, 128, 256],
        "training.weight_decay": [1e-6, 1e-5, 1e-4],
    }

    # Parameter ranges for random/optuna search
    param_ranges = {
        "training.learning_rate": ("log_uniform", 1e-5, 1e-2),
        "model.hidden_dim": ("categorical", [64, 128, 256, 512]),
        "training.weight_decay": ("log_uniform", 1e-6, 1e-4),
        "model.dropout": ("uniform", 0.0, 0.5),
    }

    # Metric to optimize (for optuna)
    optimization_metric = "val_loss"
    optimization_direction = "minimize"  # minimize or maximize
