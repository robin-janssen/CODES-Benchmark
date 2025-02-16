from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the osu2008 dataset"""

    branch_hidden_layers: int = 3
    trunk_hidden_layers: int = 2
    hidden_size: int = 378
    output_factor: int = 26
    learning_rate: float = 0.0001
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the osu2008 dataset"""

    latent_features: int = 10
    layers_factor: int = 100
    learning_rate: float = 0.002
    ode_hidden: int = 10
    ode_layer_width: int = 73
    ode_tanh_reg: bool = False
    activation: nn.Module = nn.GELU()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the osu2008 dataset"""

    hidden_size: int = 372
    num_hidden_layers: int = 4
    learning_rate: float = 0.00009  # 0.0001
    activation: nn.Module = nn.GELU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the osu2008 dataset"""

    latent_features: int = 6
    degree: int = 2
    learning_rate: float = 0.001  # 0.001
    layers_factor: int = 100
    activation: nn.Module = nn.GELU()
