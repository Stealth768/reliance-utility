"""Seed configuration utility for deterministic and reproducible experiments."""

import os
import random
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed across random, numpy, and torch (if installed).

    Parameters
    ----------
    seed : int
        Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
