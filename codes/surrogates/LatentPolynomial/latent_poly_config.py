from dataclasses import dataclass

from torch import nn


@dataclass
class LatentPolynomialBaseConfig:
    """Standard model config for LatentPolynomial"""

    latent_features: int = 5
    degree: int = 2
    coder_hidden: int = 4
    layers_factor: int = 8  # coder_layers is multiplied by this factor
    # coder_layers: list[int] = field(default_factory=lambda: [32, 16, 8])
    activation: nn.Module = nn.ReLU()
    learning_rate: float = 1e-3
