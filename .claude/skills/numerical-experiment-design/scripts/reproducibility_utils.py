"""
Reproducibility utilities for PyTorch experiments.

This module provides functions to ensure reproducible numerical experiments
by controlling all sources of randomness.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.

    This function controls randomness from:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CuDNN operations

    Args:
        seed: Random seed to use (integer)
        deterministic: If True, use deterministic algorithms (may be slower but reproducible)

    Example:
        >>> set_seed(42, deterministic=True)
        >>> # Now all random operations will be reproducible
    """
    # Python random module
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    if deterministic:
        # CuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Use deterministic algorithms where available (PyTorch >= 1.8)
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Enable cudnn benchmarking for better performance (non-deterministic)
        torch.backends.cudnn.benchmark = True

    print(f"✓ Random seed set to {seed}")
    print(f"✓ Deterministic mode: {deterministic}")


def get_environment_info() -> Dict[str, Any]:
    """
    Collect environment information for reproducibility.

    Returns:
        Dictionary containing:
        - Python version
        - PyTorch version
        - NumPy version
        - CUDA availability and version
        - GPU information
        - Other relevant environment details
    """
    import platform
    import sys

    env_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        env_info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i)
                         for i in range(torch.cuda.device_count())],
            'gpu_memory': [torch.cuda.get_device_properties(i).total_memory
                          for i in range(torch.cuda.device_count())]
        })

    return env_info


def save_environment(output_path: Path):
    """
    Save environment information to JSON file.

    Args:
        output_path: Path where environment.json will be saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_info = get_environment_info()

    with open(output_path, 'w') as f:
        json.dump(env_info, f, indent=2)

    print(f"✓ Environment saved to {output_path}")


def print_environment():
    """Print environment information to console."""
    env_info = get_environment_info()

    print("\n" + "="*50)
    print("ENVIRONMENT INFORMATION")
    print("="*50)
    print(f"Python: {env_info['python_version']}")
    print(f"Platform: {env_info['platform']}")
    print(f"PyTorch: {env_info['pytorch_version']}")
    print(f"NumPy: {env_info['numpy_version']}")
    print(f"CUDA Available: {env_info['cuda_available']}")

    if env_info['cuda_available']:
        print(f"CUDA Version: {env_info['cuda_version']}")
        print(f"cuDNN Version: {env_info['cudnn_version']}")
        print(f"GPU Count: {env_info['gpu_count']}")
        for i, name in enumerate(env_info['gpu_names']):
            memory_gb = env_info['gpu_memory'][i] / (1024**3)
            print(f"  GPU {i}: {name} ({memory_gb:.1f} GB)")

    print("="*50 + "\n")


def seed_worker(worker_id: int):
    """
    Seed worker for DataLoader reproducibility.

    Use this as worker_init_fn in PyTorch DataLoader to ensure
    reproducible data loading across workers.

    Args:
        worker_id: Worker ID provided by DataLoader

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=seed_worker,
        ...     generator=torch.Generator().manual_seed(42)
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def verify_reproducibility(func, seed: int = 42, n_runs: int = 2):
    """
    Verify that a function produces reproducible results.

    Args:
        func: Function to test (should return a torch.Tensor or numpy array)
        seed: Seed to use for testing
        n_runs: Number of runs to compare (default: 2)

    Returns:
        True if results are identical across runs, False otherwise

    Example:
        >>> def train_step():
        ...     # Your training code
        ...     return loss
        >>> is_reproducible = verify_reproducibility(train_step, seed=42)
    """
    results = []

    for i in range(n_runs):
        set_seed(seed, deterministic=True)
        result = func()

        if isinstance(result, torch.Tensor):
            result = result.detach().cpu()

        results.append(result)

    # Compare all results
    all_equal = True
    for i in range(1, len(results)):
        if isinstance(results[0], torch.Tensor):
            if not torch.allclose(results[0], results[i]):
                all_equal = False
                break
        else:
            if not np.allclose(results[0], results[i]):
                all_equal = False
                break

    if all_equal:
        print(f"✓ Function is reproducible across {n_runs} runs")
    else:
        print(f"✗ Function is NOT reproducible")
        print(f"  Results differ between runs")

    return all_equal


def create_reproducibility_report(output_dir: Path, config: Dict, seed: int):
    """
    Create a comprehensive reproducibility report.

    Args:
        output_dir: Directory to save the report
        config: Experiment configuration dictionary
        seed: Random seed used

    Creates:
        - environment.json: Environment information
        - config.json: Experiment configuration
        - reproducibility_checklist.txt: Checklist of reproducibility measures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save environment
    save_environment(output_dir / 'environment.json')

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create checklist
    checklist = f"""REPRODUCIBILITY CHECKLIST
========================

✓ Random seed set: {seed}
✓ Environment documented: See environment.json
✓ Configuration saved: See config.json
✓ PyTorch version: {torch.__version__}

To reproduce these results:
1. Install exact package versions from requirements.txt
2. Use the same hardware (GPU model matters for some operations)
3. Load configuration from config.json
4. Set random seed to {seed}
5. Enable deterministic mode (torch.backends.cudnn.deterministic = True)

Notes:
- Results are reproducible on the same hardware with same software versions
- Cross-platform reproducibility (CPU vs GPU, different GPU models) is not guaranteed
- Some operations may not have deterministic implementations

Environment Details:
-------------------
{json.dumps(get_environment_info(), indent=2)}
"""

    with open(output_dir / 'reproducibility_checklist.txt', 'w') as f:
        f.write(checklist)

    print(f"✓ Reproducibility report saved to {output_dir}")


if __name__ == '__main__':
    # Example usage
    print("Reproducibility Utilities")
    print("-" * 50)

    # Set seed
    set_seed(42, deterministic=True)

    # Print environment
    print_environment()

    # Example: verify reproducibility
    def example_function():
        return torch.randn(10)

    verify_reproducibility(example_function, seed=42, n_runs=3)
