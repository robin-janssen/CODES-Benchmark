from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the osu2008 dataset"""

    branch_hidden_layers: int = 8
    trunk_hidden_layers: int = 8
    hidden_size: int = 280
    output_factor: int = 30
    learning_rate: float = 3e-4
    activation: nn.Module = nn.GELU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the osu2008 dataset"""

    latent_features: int = 7
    layers_factor: int = 46
    learning_rate: float = 0.003
    ode_activation: nn.Module = nn.Tanh()
    ode_tanh_reg: bool = True
    coder_activation: nn.Module = nn.Tanh()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the osu2008 dataset"""

    hidden_size: int = 80
    num_hidden_layers: int = 2
    learning_rate: float = 1.5e-5
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the osu2008 dataset"""

    latent_features: int = 10
    degree: int = 1
    learning_rate: float = 5e-4
    layers_factor: int = 50
    coder_activation: nn.Module = nn.GELU()
