"""
Utility functions for numerical experiments.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed to use
        deterministic: If True, use deterministic algorithms (may be slower)

    Note:
        Setting deterministic=True ensures reproducibility but may impact performance.
        Some operations may not have deterministic implementations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms (PyTorch >= 1.8)
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Path,
    **kwargs
):
    """
    Save model checkpoint with all necessary information for resumption.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics to save
        filepath: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into

    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def get_device(use_cuda: bool = True) -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data: Dict, filepath: Path):
    """Save dictionary to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Dict:
    """Load dictionary from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)
