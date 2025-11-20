"""
Generator network for DCGAN-style Pokémon generation.
Takes a 100-dimensional noise vector and generates a 64x64x3 RGB image.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN-style Generator network.
    
    Architecture:
    - Input: z ∈ R^100 (Gaussian noise vector)
    - Linear: 100 → 4×4×512 (reshape to 4×4×512 feature map)
    - 4 TransposeConv2d blocks with BatchNorm and ReLU
    - Final layer with Tanh activation
    - Output: 64×64×3 RGB image with values in [-1, 1]
    """
    
    def __init__(self, nz=100, ngf=512, nc=3):
        """
        Args:
            nz: Size of input noise vector (default: 100)
            ngf: Number of generator features in first layer (default: 512)
            nc: Number of channels in output image (default: 3 for RGB)
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        # Linear layer: 100 -> 4×4×512
        self.linear = nn.Sequential(
            nn.Linear(nz, 4 * 4 * ngf),
            nn.BatchNorm1d(4 * 4 * ngf),
            nn.ReLU(True)
        )
        
        # Transpose convolution layers
        self.main = nn.Sequential(
            # TransposeConv2d: 4×4×512 → 8×8×256
            nn.ConvTranspose2d(ngf, ngf // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            
            # TransposeConv2d: 8×8×256 → 16×16×128
            nn.ConvTranspose2d(ngf // 2, ngf // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            
            # TransposeConv2d: 16×16×128 → 32×32×64
            nn.ConvTranspose2d(ngf // 4, ngf // 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf // 8),
            nn.ReLU(True),
            
            # TransposeConv2d: 32×32×64 → 64×64×3
            nn.ConvTranspose2d(ngf // 8, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output values in [-1, 1]
        )
    
    def forward(self, input):
        """
        Forward pass through generator.
        
        Args:
            input: Noise tensor of shape (batch_size, nz)
            
        Returns:
            Generated image tensor of shape (batch_size, nc, 64, 64)
        """
        # Linear layer and reshape
        x = self.linear(input)
        x = x.view(-1, self.ngf, 4, 4)  # Reshape to (batch, 512, 4, 4)
        
        # Pass through transpose convolutions
        x = self.main(x)
        
        return x


def test_generator():
    """Test function to verify generator architecture."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG = Generator(nz=100, ngf=512, nc=3).to(device)
    
    # Create random noise
    noise = torch.randn(4, 100, device=device)
    
    # Generate images
    with torch.no_grad():
        fake_images = netG(noise)
    
    print(f"Generator output shape: {fake_images.shape}")
    print(f"Expected shape: (4, 3, 64, 64)")
    print(f"Min value: {fake_images.min().item():.3f}, Max value: {fake_images.max().item():.3f}")
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    
    assert fake_images.shape == (4, 3, 64, 64), f"Expected (4, 3, 64, 64), got {fake_images.shape}"
    assert -1.0 <= fake_images.min().item() <= 1.0, "Output should be in [-1, 1] range"
    assert -1.0 <= fake_images.max().item() <= 1.0, "Output should be in [-1, 1] range"
    print("✓ Generator test passed!")


if __name__ == "__main__":
    test_generator()

