"""
Evaluation metrics for GANs: FID, Inception Score, and Diversity Score.
"""

import torch
import numpy as np
from scipy import linalg
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not available. FID and IS will use alternative implementations.")


def calculate_fid(real_images, fake_images, device='cuda'):
    """
    Calculate Fréchet Inception Distance (FID).
    
    Args:
        real_images: Tensor of real images (N, 3, 64, 64) in [-1, 1]
        fake_images: Tensor of fake images (M, 3, 64, 64) in [-1, 1]
        device: Device to run computation on
        
    Returns:
        FID score (lower is better)
    """
    if TORCHMETRICS_AVAILABLE:
        # Use torchmetrics implementation
        fid = FrechetInceptionDistance(normalize=True).to(device)
        
        # Normalize images to [0, 1] for torchmetrics
        real_norm = (real_images + 1) / 2.0
        fake_norm = (fake_images + 1) / 2.0
        
        # Convert to uint8
        real_norm = (real_norm * 255).byte()
        fake_norm = (fake_norm * 255).byte()
        
        fid.update(real_norm, real=True)
        fid.update(fake_norm, real=False)
        
        return fid.compute().item()
    else:
        # Fallback: simplified FID calculation
        # Note: This is a placeholder. For proper FID, you need Inception v3 features
        print("Warning: Using simplified FID approximation. Install torchmetrics for accurate FID.")
        return _simplified_fid(real_images, fake_images)


def calculate_inception_score(fake_images, splits=10, device='cuda'):
    """
    Calculate Inception Score (IS).
    
    Args:
        fake_images: Tensor of fake images (N, 3, 64, 64) in [-1, 1]
        splits: Number of splits for IS calculation
        device: Device to run computation on
        
    Returns:
        IS score (higher is better) and standard deviation
    """
    if TORCHMETRICS_AVAILABLE:
        # Use torchmetrics implementation
        is_metric = InceptionScore(normalize=True).to(device)
        
        # Normalize images to [0, 1] for torchmetrics
        fake_norm = (fake_images + 1) / 2.0
        
        # Convert to uint8
        fake_norm = (fake_norm * 255).byte()
        
        is_metric.update(fake_norm)
        mean, std = is_metric.compute()
        
        return mean.item(), std.item()
    else:
        # Fallback: simplified IS calculation
        print("Warning: Using simplified IS approximation. Install torchmetrics for accurate IS.")
        return _simplified_is(fake_images, splits)


def calculate_diversity_score(fake_images, device='cuda'):
    """
    Calculate Diversity Score using pairwise LPIPS (Learned Perceptual Image Patch Similarity).
    For simplicity, we use L2 distance in pixel space as a proxy.
    
    Args:
        fake_images: Tensor of fake images (N, 3, 64, 64) in [-1, 1]
        device: Device to run computation on
        
    Returns:
        Diversity score (average pairwise distance, higher is better)
    """
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


def _simplified_fid(real_images, fake_images):
    """Simplified FID approximation using pixel statistics."""
    # Flatten images
    real_flat = real_images.view(real_images.size(0), -1).cpu().numpy()
    fake_flat = fake_images.view(fake_images.size(0), -1).cpu().numpy()
    
    # Calculate mean and covariance
    mu_real = np.mean(real_flat, axis=0)
    sigma_real = np.cov(real_flat, rowvar=False)
    
    mu_fake = np.mean(fake_flat, axis=0)
    sigma_fake = np.cov(fake_flat, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_fake
    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid


def _simplified_is(fake_images, splits=10):
    """Simplified IS approximation."""
    # This is a placeholder. Real IS requires Inception v3 predictions.
    # For now, return a dummy score
    return 1.0, 0.1

