import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, use_spectral_norm=False):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.use_spectral_norm = use_spectral_norm
        
        # Helper function to optionally apply spectral normalization
        def maybe_spectral_norm(module):
            if use_spectral_norm:
                return nn.utils.spectral_norm(module)
            return module
        
        # Build convolutional layers
        conv_layers = []
        
        # Conv2d: 64×64×3 → 32×32×64
        conv_layers.append(maybe_spectral_norm(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        ))
        conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Conv2d: 32×32×64 → 16×16×128
        conv_layers.append(maybe_spectral_norm(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        ))
        # Skip BatchNorm when using spectral normalization (they can conflict)
        if not use_spectral_norm:
            conv_layers.append(nn.BatchNorm2d(ndf * 2))
        conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Conv2d: 16×16×128 → 8×8×256
        conv_layers.append(maybe_spectral_norm(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        ))
        if not use_spectral_norm:
            conv_layers.append(nn.BatchNorm2d(ndf * 4))
        conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Conv2d: 8×8×256 → 4×4×512
        conv_layers.append(maybe_spectral_norm(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        ))
        if not use_spectral_norm:
            conv_layers.append(nn.BatchNorm2d(ndf * 8))
        conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.main = nn.Sequential(*conv_layers)
        
        # Flatten → Linear: 4×4×512 → 1
        # For LSGAN, we don't use Sigmoid (outputs raw scores)
        # For BCE, we use Sigmoid
        fc_layers = [
            nn.Flatten(),
            maybe_spectral_norm(nn.Linear(ndf * 8 * 4 * 4, 1))
        ]
        self.fc = nn.Sequential(*fc_layers)
        self.use_sigmoid = True  # Default to BCE/Sigmoid, can be changed for LSGAN
    
    def forward(self, input):
        x = self.main(input)
        x = self.fc(x)
        if self.use_sigmoid:
            return torch.sigmoid(x)
        return x

def test_discriminator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing Discriminator without spectral normalization...")
    netD = Discriminator(nc=3, ndf=64, use_spectral_norm=False).to(device)
    
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
    print("Discriminator test passed!")
    
    print("\nTesting Discriminator with spectral normalization...")
    netD_sn = Discriminator(nc=3, ndf=64, use_spectral_norm=True).to(device)
    
    with torch.no_grad():
        output_sn = netD_sn(real_images)
    
    print(f"Discriminator (SN) output shape: {output_sn.shape}")
    print(f"Discriminator (SN) parameters: {sum(p.numel() for p in netD_sn.parameters()):,}")
    assert output_sn.shape == (4, 1), f"Expected (4, 1), got {output_sn.shape}"
    print("Discriminator with spectral normalization test passed!")


if __name__ == "__main__":
    test_discriminator()