# PyTorch Reproducibility Best Practices

This document outlines comprehensive best practices for ensuring reproducibility in PyTorch-based numerical experiments.

## Core Principles

Reproducibility in PyTorch requires controlling three main sources of randomness:
1. **Software-level randomness** (Python, NumPy, PyTorch)
2. **Hardware-level randomness** (CUDA operations)
3. **Data loading randomness** (DataLoader workers)

## 1. Setting Random Seeds

### Basic Seed Setting

```python
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
```

### Environment Variable

Set `PYTHONHASHSEED` before Python starts:

```bash
export PYTHONHASHSEED=0
python train.py
```

Or in Python (must be before imports):

```python
import os
os.environ['PYTHONHASHSEED'] = '0'
```

## 2. CuDNN Deterministic Operations

CuDNN (CUDA Deep Neural Network library) uses non-deterministic algorithms by default for performance.

```python
import torch

# For reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For performance (non-deterministic)
torch.backends.cudnn.benchmark = True
```

### Performance vs. Reproducibility Trade-off

- `deterministic=True, benchmark=False`: Fully reproducible, slower
- `deterministic=False, benchmark=True`: Faster, non-reproducible
- Speedup from `benchmark=True` can be 20-50% for convolutional networks

## 3. Deterministic Algorithms

PyTorch 1.8+ provides `torch.use_deterministic_algorithms`:

```python
torch.use_deterministic_algorithms(True, warn_only=True)
```

This ensures operations use deterministic algorithms when available. `warn_only=True` issues warnings instead of errors for operations without deterministic implementations.

### Operations Without Deterministic Implementations

Some operations don't have deterministic implementations:
- `torch.nn.functional.interpolate` (bilinear/bicubic)
- Certain sparse tensor operations
- Some indexing operations

Check PyTorch documentation for current status.

## 4. DataLoader Reproducibility

### Worker Initialization

DataLoader workers need proper seeding:

```python
def seed_worker(worker_id):
    """Seed worker for reproducible data loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Use in DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=torch.Generator().manual_seed(42)
)
```

### Drop Last Batch

Enable `drop_last=True` to avoid inconsistent final batch sizes:

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    drop_last=True,  # Ensures consistent batch sizes
    shuffle=True
)
```

## 5. Model Initialization

Control model initialization by setting the seed before creating the model:

```python
set_seed(42)
model = MyModel()  # Initialization will be deterministic
```

## 6. Environment Documentation

### Track Environment Information

```python
import torch
import json

def save_environment(filepath):
    env_info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    }
    with open(filepath, 'w') as f:
        json.dump(env_info, f, indent=2)
```

### Requirements Pinning

Pin exact versions in `requirements.txt`:

```txt
torch==2.0.1
numpy==1.24.3
```

Better: use `pip freeze > requirements.txt` after environment setup.

## 7. Complete Reproducibility Function

```python
def set_reproducibility(seed: int, deterministic: bool = True):
    """
    Set all random seeds and configure PyTorch for reproducibility.

    Args:
        seed: Random seed
        deterministic: If True, use deterministic operations (slower)
    """
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
```

## 8. Important Caveats

### Cross-Platform Reproducibility

Completely reproducible results are **NOT guaranteed** across:
- Different PyTorch releases
- Different hardware (CPU vs GPU, different GPU models)
- Different operating systems
- Different CUDA versions

### What IS Reproducible

Results **ARE reproducible** when:
- Same hardware
- Same software versions (PyTorch, CUDA, cuDNN)
- Same operating system
- Proper seeding as described above

### Documentation Requirements

For full reproducibility, document:
1. Exact PyTorch version (`torch.__version__`)
2. CUDA version if using GPU
3. Hardware used (GPU model, CPU)
4. Operating system
5. All hyperparameters
6. Random seeds used
7. Data preprocessing steps

## 9. Testing Reproducibility

Verify reproducibility with multiple runs:

```python
# Run 1
set_seed(42)
result1 = train_model()

# Run 2
set_seed(42)
result2 = train_model()

assert torch.allclose(result1, result2), "Results not reproducible!"
```

## 10. Debugging Non-Reproducibility

If results aren't reproducible:

1. Check all seeds are set before any random operations
2. Verify `worker_init_fn` in DataLoader
3. Check for operations without deterministic implementations
4. Ensure model initialization happens after seed setting
5. Verify no async operations or multi-threading without proper seeding
6. Check for operations that depend on memory layout (use `.contiguous()`)

## References

- PyTorch Reproducibility Documentation: https://pytorch.org/docs/stable/notes/randomness.html
- PyTorch Deterministic Operations: https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
