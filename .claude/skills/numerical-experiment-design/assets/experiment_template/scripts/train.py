"""
Main training script for numerical experiments.

This script demonstrates best practices for reproducible PyTorch experiments:
- Comprehensive seed setting for reproducibility
- Proper logging and checkpointing
- Environment tracking
- Configuration management
"""

import argparse
import json
import sys
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed to use
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Use deterministic algorithms where available (PyTorch >= 1.8)
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def save_environment(output_dir: Path):
    """Save environment information for reproducibility."""
    env_info = {
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        env_info["gpu_name"] = torch.cuda.get_device_name(0)

    with open(output_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)


def create_experiment_dir(config) -> Path:
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = config.results_dir / config.experiment_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Logging
        if batch_idx % config.logging["log_interval"] == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.6f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", type=str, default="configs/base_config.py",
                        help="Path to config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    args = parser.parse_args()

    # Load config
    # Note: In practice, use importlib or hydra for config loading
    from configs.base_config import Config
    config = Config()

    # Override seed if provided
    if args.seed is not None:
        config.seed = args.seed

    # Set seeds for reproducibility
    set_seed(config.seed, config.deterministic)

    # Create experiment directory
    exp_dir = create_experiment_dir(config)
    print(f"Experiment directory: {exp_dir}")

    # Save environment and config
    save_environment(exp_dir)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Initialize tensorboard
    if config.logging["use_tensorboard"]:
        writer = SummaryWriter(log_dir=exp_dir / "tensorboard")
    else:
        writer = None

    # TODO: Initialize your model, data loaders, optimizer, etc.
    # model = YourModel(**config.model).to(config.device)
    # train_loader = create_dataloader(config, train=True)
    # val_loader = create_dataloader(config, train=False)
    # optimizer = create_optimizer(model, config)
    # criterion = nn.MSELoss()

    print(f"Using device: {config.device}")
    print(f"Random seed: {config.seed}")
    print(f"Deterministic mode: {config.deterministic}")

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(config.training["num_epochs"]):
        # TODO: Implement training
        # train_loss = train_epoch(model, train_loader, optimizer, criterion,
        #                          config.device, epoch, writer, config)

        # Validation
        # if epoch % config.validation["val_interval"] == 0:
        #     val_loss = validate(model, val_loader, criterion, config.device)
        #
        #     if writer:
        #         writer.add_scalar("val/loss", val_loss, epoch)
        #
        #     # Save best model
        #     if val_loss < best_val_loss:
        #         best_val_loss = val_loss
        #         torch.save({
        #             "epoch": epoch,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "val_loss": val_loss,
        #             "config": config.to_dict(),
        #         }, exp_dir / "best_model.pt")

        pass

    if writer:
        writer.close()

    print(f"Training complete! Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
