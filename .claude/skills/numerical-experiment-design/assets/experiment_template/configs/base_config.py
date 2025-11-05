"""
Base configuration for numerical experiments.

This configuration ensures reproducibility and provides a clear structure
for hyperparameter management.
"""

import torch
from pathlib import Path

class Config:
    """Base experiment configuration."""

    # Experiment metadata
    experiment_name = "base_experiment"
    description = "Brief description of this experiment"

    # Reproducibility
    seed = 42
    deterministic = True  # Enable PyTorch deterministic operations

    # Paths
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    results_dir = root_dir / "results"
    log_dir = root_dir / "logs"

    # Model hyperparameters
    model = {
        "name": "your_model",
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.1,
    }

    # Training hyperparameters
    training = {
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "optimizer": "adam",  # adam, sgd, adamw
        "scheduler": "cosine",  # cosine, step, exponential, none
    }

    # Data loader settings
    dataloader = {
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": True,  # Ensures consistent batch sizes for reproducibility
    }

    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    logging = {
        "log_interval": 10,  # Log every N iterations
        "save_interval": 10,  # Save checkpoint every N epochs
        "use_tensorboard": True,
        "use_wandb": False,
    }

    # Validation
    validation = {
        "val_interval": 1,  # Validate every N epochs
        "early_stopping_patience": 10,
    }

    def to_dict(self):
        """Convert config to dictionary for logging."""
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "model": self.model,
            "training": self.training,
            "dataloader": self.dataloader,
            "device": self.device,
            "logging": self.logging,
            "validation": self.validation,
        }
