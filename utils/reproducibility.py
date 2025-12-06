import random
import numpy as np
import torch
import os


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set random seed to {seed}.")

def make_deterministic():
    """Make PyTorch operations deterministic (may reduce performance)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = '0'
    print("Enabled deterministic operations.")