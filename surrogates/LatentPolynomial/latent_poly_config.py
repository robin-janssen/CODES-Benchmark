from dataclasses import dataclass, field

from torch import nn

@dataclass
class LatentPolynomialConfigOSU:
    "Model Config for LatentPolynomial model for OUS_2008 dataset"
    in_features: int = 29
    latent_features: int = 5
    degree: int = 2
    latent_dim: int = 5
    coder_hidden: int = 4
    coder_layers: list[int] = field(default_factory=lambda: [32, 16, 8])
    coder_activation: nn.Module = nn.ReLU()
    
    learning_rate: float = 1e-3
    epochs: int = 50000

    device: str = "cuda:0"
    batch_size: int = 256
