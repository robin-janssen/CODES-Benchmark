from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the simple_ode dataset"""

    branch_hidden_layers: int = 6
    trunk_hidden_layers: int = 4
    hidden_size: int = 347
    output_factor: int = 90
    learning_rate: float = 1.2e-5
    activation: nn.Module = nn.ReLU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the simple_ode dataset"""

    latent_features: int = 9
    coder_layers: int = 4
    coder_width: int = 230
    learning_rate: float = 0.0004
    ode_layers: int = 5
    ode_width: int = 150
    ode_tanh_reg: bool = True
    activation: nn.Module = nn.LeakyReLU()
    model_version: str = "v2"


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the simple_ode dataset"""

    hidden_size: int = 800
    num_hidden_layers: int = 8
    learning_rate: float = 3e-5
    activation: nn.Module = nn.GELU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the simple_ode dataset"""

    latent_features: int = 6
    degree: int = 2
    learning_rate: float = 0.002
    coder_layers: int = 4
    coder_width: int = 230
    activation: nn.Module = nn.LeakyReLU()
