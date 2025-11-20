"""
Discriminator network for DCGAN-style Pokémon generation.
Takes a 64x64x3 RGB image and outputs a probability score (real vs. fake).
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN-style Discriminator network.
    
    Architecture:
    - Input: 64×64×3 RGB image
    - 4 Conv2d blocks with LeakyReLU and BatchNorm
    - Strided convolutions instead of pooling
    - Flatten → Linear → Sigmoid
    - Output: Probability score (real vs. fake)
    """
    
    def __init__(self, nc=3, ndf=64):
        """
        Args:
            nc: Number of channels in input image (default: 3 for RGB)
            ndf: Number of discriminator features in first layer (default: 64)
        """
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        
        self.main = nn.Sequential(
            # Conv2d: 64×64×3 → 32×32×64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv2d: 32×32×64 → 16×16×128
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv2d: 16×16×128 → 8×8×256
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv2d: 8×8×256 → 4×4×512
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Flatten → Linear: 4×4×512 → 1
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf * 8 * 4 * 4, 1),
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, input):
        """
        Forward pass through discriminator.
        
        Args:
            input: Image tensor of shape (batch_size, nc, 64, 64)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        x = self.main(input)
        x = self.fc(x)
        return x


def test_discriminator():
    """Test function to verify discriminator architecture."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netD = Discriminator(nc=3, ndf=64).to(device)
    
    # Create random images
    real_images = torch.randn(4, 3, 64, 64, device=device)
    
    # Discriminate
    with torch.no_grad():
        output = netD(real_images)
    
    print(f"Discriminator output shape: {output.shape}")
    print(f"Expected shape: (4, 1)")
    print(f"Min value: {output.min().item():.3f}, Max value: {output.max().item():.3f}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")
    
    assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"
    assert 0.0 <= output.min().item() <= 1.0, "Output should be in [0, 1] range (Sigmoid)"
    assert 0.0 <= output.max().item() <= 1.0, "Output should be in [0, 1] range (Sigmoid)"
    print("✓ Discriminator test passed!")


if __name__ == "__main__":
    test_discriminator()

