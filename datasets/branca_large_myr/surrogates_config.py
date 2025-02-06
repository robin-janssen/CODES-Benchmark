from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the osu2008 dataset"""

    branch_hidden_layers: int = 5
    trunk_hidden_layers: int = 10
    hidden_size: int = 512
    output_factor: int = 50
    learning_rate: float = 0.0001  # 0.0001
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the osu2008 dataset"""

    latent_features: int = 9
    layers_factor: int = 86
    learning_rate: float = 0.0007
    ode_hidden: int = 3
    ode_layer_width: int = 412
    ode_tanh_reg: bool = True
    activation: nn.Module = nn.ReLU()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the osu2008 dataset"""

    hidden_size: int = 1000
    num_hidden_layers: int = 6
    learning_rate: float = 0.00005
    activation: nn.Module = nn.Tanh()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the osu2008 dataset"""

    latent_features: int = 8
    degree: int = 7
    learning_rate: float = 0.0004  # 0.001
    layers_factor: int = 84
    activation: nn.Module = nn.ReLU()
