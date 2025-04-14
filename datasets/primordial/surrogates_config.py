from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the simple_ode dataset"""

    branch_hidden_layers: int = 8
    trunk_hidden_layers: int = 6
    hidden_size: int = 250
    output_factor: int = 115
    learning_rate: float = 1e-4
    activation: nn.Module = nn.Tanh()


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

    latent_features: int = 9
    degree: int = 1
    learning_rate: float = 0.0002
    coder_layers: int = 2
    coder_width: int = 264
    activation: nn.Module = nn.LeakyReLU()
