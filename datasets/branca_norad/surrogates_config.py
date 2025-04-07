from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the osu2008 dataset"""

    branch_hidden_layers: int = 7
    trunk_hidden_layers: int = 4
    hidden_size: int = 400
    output_factor: int = 62
    learning_rate: float = 1e-4
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the osu2008 dataset"""

    latent_features: int = 9
    layers_factor: int = 84
    learning_rate: float = 0.002
    ode_tanh_reg: bool = False
    activation: nn.Module = nn.GELU()
    model_version: str = "v1"


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the osu2008 dataset"""

    hidden_size: int = 29
    num_hidden_layers: int = 1
    learning_rate: float = 1e-4
    activation: nn.Module = nn.ReLU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the osu2008 dataset"""

    latent_features: int = 16
    degree: int = 1
    learning_rate: float = 3e-4
    layers_factor: int = 92
    activation: nn.Module = nn.GELU()
