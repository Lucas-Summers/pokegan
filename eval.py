"""
Evaluation script for DCGAN Pokémon generator.
Evaluates model on held-out test data and generates metrics.
"""

import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from models import Generator, Discriminator
from data import PokemonDataset
from utils.reproducibility import set_seed
from utils.visualization import save_image_grid, plot_confusion_matrix
from utils.metrics import calculate_fid, calculate_inception_score, calculate_diversity_score


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DCGAN for Pokémon generation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test_dir', type=str, default=None,
                       help='Path to test data directory (overrides config)')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate for evaluation')
    parser.add_argument('--output_dir', type=str, default='eval_outputs',
                       help='Directory to save evaluation outputs')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, config, device):
    """Load generator and discriminator from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create models
    netG = Generator(
        nz=config['model']['nz'],
        ngf=config['model']['ngf'],
        nc=config['model']['nc']
    ).to(device)
    
    netD = Discriminator(
        nc=config['model']['nc'],
        ndf=config['model']['ndf']
    ).to(device)
    
    # Load weights
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    
    netG.eval()
    netD.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        print(f"Best FID: {checkpoint['metrics'].get('best_fid', 'N/A')}")
    
    return netG, netD


def generate_samples(netG, n_samples, nz, device, batch_size=64):
    """Generate fake samples."""
    print(f"Generating {n_samples} samples...")
    fake_images = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_size_actual = min(batch_size, n_samples - i)
            noise = torch.randn(batch_size_actual, nz, device=device)
            fake_batch = netG(noise)
            fake_images.append(fake_batch.cpu())
    
    fake_images = torch.cat(fake_images, dim=0)
    return fake_images


def evaluate_discriminator(netD, real_images, fake_images, device):
    """Evaluate discriminator performance."""
    print("Evaluating discriminator...")
    
    real_scores = []
    fake_scores = []
    
    batch_size = 64
    with torch.no_grad():
        # Real images
        for i in range(0, len(real_images), batch_size):
            batch = real_images[i:i+batch_size].to(device)
            scores = netD(batch)
            real_scores.append(scores.cpu())
        
        # Fake images
        for i in range(0, len(fake_images), batch_size):
            batch = fake_images[i:i+batch_size].to(device)
            scores = netD(batch)
            fake_scores.append(scores.cpu())
    
    real_scores = torch.cat(real_scores, dim=0)
    fake_scores = torch.cat(fake_scores, dim=0)
    
    return real_scores, fake_scores


def main():
    """Main evaluation function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Set reproducibility
    set_seed(config['training']['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    netG, netD = load_model(args.checkpoint, config, device)
    
    # Load test data
    test_dir = args.test_dir if args.test_dir else config['data'].get('test_dir', config['data']['train_dir'])
    test_dataset = PokemonDataset(root_dir=test_dir, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Collect real images
    print("Collecting real images...")
    real_images = []
    for batch in test_loader:
        real_images.append(batch)
    real_images = torch.cat(real_images, dim=0)
    
    # Limit real images for comparison (use same number as generated)
    n_eval = min(args.n_samples, len(real_images))
    real_images = real_images[:n_eval]
    
    print(f"Using {n_eval} real images for evaluation")
    
    # Generate fake samples
    fake_images = generate_samples(netG, args.n_samples, config['model']['nz'], device)
    
    # Save sample grids
    print("Saving sample images...")
    save_image_grid(real_images[:64], os.path.join(args.output_dir, 'real_samples.png'))
    save_image_grid(fake_images[:64], os.path.join(args.output_dir, 'fake_samples.png'))
    
    # Calculate metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    # FID
    print("\nCalculating FID...")
    fid = calculate_fid(real_images.to(device), fake_images.to(device), device=device)
    print(f"FID Score: {fid:.4f} (lower is better)")
    
    # Inception Score
    print("\nCalculating Inception Score...")
    is_mean, is_std = calculate_inception_score(fake_images.to(device), device=device)
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f} (higher is better)")
    
    # Diversity Score
    print("\nCalculating Diversity Score...")
    diversity = calculate_diversity_score(fake_images.to(device), device=device)
    print(f"Diversity Score: {diversity:.4f} (higher is better)")
    
    # Discriminator evaluation
    print("\nEvaluating discriminator...")
    real_scores, fake_scores = evaluate_discriminator(netD, real_images, fake_images, device)
    
    # Confusion matrix
    plot_confusion_matrix(real_scores, fake_scores, 
                         save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Discriminator statistics
    print(f"\nDiscriminator Statistics:")
    print(f"  Real images - Mean score: {real_scores.mean().item():.4f}, "
          f"Std: {real_scores.std().item():.4f}")
    print(f"  Fake images - Mean score: {fake_scores.mean().item():.4f}, "
          f"Std: {fake_scores.std().item():.4f}")
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("EVALUATION METRICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"FID Score: {fid:.4f}\n")
        f.write(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}\n")
        f.write(f"Diversity Score: {diversity:.4f}\n\n")
        f.write("Discriminator Statistics:\n")
        f.write(f"  Real images - Mean: {real_scores.mean().item():.4f}, "
                f"Std: {real_scores.std().item():.4f}\n")
        f.write(f"  Fake images - Mean: {fake_scores.mean().item():.4f}, "
                f"Std: {fake_scores.std().item():.4f}\n")
    
    print(f"\nMetrics saved to {metrics_file}")
    print(f"\nEvaluation complete! Outputs saved to {args.output_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"FID: {fid:.4f}")
    print(f"IS: {is_mean:.4f} ± {is_std:.4f}")
    print(f"Diversity: {diversity:.4f}")


if __name__ == '__main__':
    main()

