import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

def calculate_fid(real_images, fake_images, device='cuda'):
    """Calculate Fréchet Inception Distance (FID)."""
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # Normalize images to [0, 1] and convert to uint8 for torchmetrics
    real_norm = (real_images + 1) / 2.0
    fake_norm = (fake_images + 1) / 2.0
    real_norm = (real_norm * 255).byte()
    fake_norm = (fake_norm * 255).byte()
    
    fid.update(real_norm, real=True)
    fid.update(fake_norm, real=False)
    
    return fid.compute().item()

def calculate_inception_score(fake_images, splits=10, device='cuda'):
    """Calculate Inception Score (IS)."""
    is_metric = InceptionScore(normalize=True).to(device)
    
    # Normalize images to [0, 1] and convert to uint8 for torchmetrics
    fake_norm = (fake_images + 1) / 2.0
    fake_norm = (fake_norm * 255).byte()
    
    is_metric.update(fake_norm)
    mean, std = is_metric.compute()
    
    return mean.item(), std.item()

def calculate_diversity_score(fake_images, device='cuda'):
    """Calculate Diversity Score (DS)."""
    # Flatten images
    fake_flat = fake_images.view(fake_images.size(0), -1)
    
    # Calculate pairwise L2 distances
    n = fake_flat.size(0)
    distances = []
    
    # Sample pairs to avoid O(n^2) computation
    n_samples = min(1000, n * (n - 1) // 2)
    indices = torch.randperm(n * (n - 1) // 2, device=device)[:n_samples]
    
    for idx in indices:
        i = idx // (n - 1)
        j = idx % (n - 1)
        if j >= i:
            j += 1
        dist = torch.norm(fake_flat[i] - fake_flat[j]).item()
        distances.append(dist)
    
    diversity = np.mean(distances)
    return diversity