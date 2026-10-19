"""
Training utilities.
"""

import random
import numpy as np

import torch


def set_seed(seed: int) -> None:
    """
    Set seeds for packages. Runs are not fully deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_confusion(cm: np.ndarray) -> str:
    """
    Format the confusion matrix (activity prediction) printed during training.
    """
    rows = ["true\\pred " + " ".join(f"{i:>5d}" for i in range(cm.shape[1]))]
    for i, row in enumerate(cm):
        rows.append(f"{i:>9d} " + " ".join(f"{v:>5d}" for v in row))
    return "\n".join(rows)
