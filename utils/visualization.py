import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_image_grid(images, filepath, nrow=8, normalize=True, value_range=(-1, 1)):
    """Save a grid of images."""
    if isinstance(images, list):
        images = torch.stack(images)
    
    # Denormalize if needed
    if normalize and value_range == (-1, 1):
        images = (images + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
        images = torch.clamp(images, 0, 1)
    
    # Create grid
    grid = make_grid(images, nrow=nrow, normalize=False, padding=2)
    
    # Convert to numpy and save
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved image grid to {filepath}.")


def plot_training_curves(train_losses, val_losses, train_metrics=None, val_metrics=None, 
                         save_path='training_curves.png'):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    axes[0].plot(train_losses, label='Train Loss', alpha=0.7)
    if val_losses:
        axes[0].plot(val_losses, label='Val Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics if provided
    if train_metrics or val_metrics:
        for metric_name, values in (train_metrics or {}).items():
            axes[1].plot(values, label=f'Train {metric_name}', alpha=0.7)
        for metric_name, values in (val_metrics or {}).items():
            axes[1].plot(values, label=f'Val {metric_name}', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Training Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_confusion_matrix(real_scores, fake_scores, save_path='confusion_matrix.png', threshold=0.5):
    """Plot confusion matrix for discriminator predictions."""
    # Convert to numpy
    if isinstance(real_scores, torch.Tensor):
        real_scores = real_scores.cpu().numpy()
    if isinstance(fake_scores, torch.Tensor):
        fake_scores = fake_scores.cpu().numpy()
    
    # Classify
    real_pred = (real_scores >= threshold).astype(int)
    fake_pred = (fake_scores < threshold).astype(int)
    
    # Calculate confusion matrix
    tp = np.sum(real_pred == 1)
    fn = np.sum(real_pred == 0)
    fp = np.sum(fake_pred == 1)
    tn = np.sum(fake_pred == 0)
    
    cm = np.array([[tp, fn], [fp, tn]])
    
    # Plot and add text annotations
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set(xticks=np.arange(2),
           yticks=np.arange(2),
           xticklabels=['Predicted Real', 'Predicted Fake'],
           yticklabels=['Actual Real', 'Actual Fake'],
           title='Discriminator Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}.")
    print("Accuracy statistics:")
    print(f"  Accuracy: {(tp + tn) / (tp + tn + fp + fn):.3f}")
    print(f"  Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.3f}")
    print(f"  Recall: {tp / (tp + fn) if (tp + fn) > 0 else 0:.3f}")
    print(f"  F1 Score: {2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.3f}")
