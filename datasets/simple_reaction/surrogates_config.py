from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the simple_reaction dataset"""

    branch_hidden_layers: int = 6
    trunk_hidden_layers: int = 2
    hidden_size: int = 672
    output_factor: int = 58
    learning_rate: float = 1.8e-5
    activation: nn.Module = nn.GELU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the simple_reaction dataset"""

    latent_features: int = 10
    layers_factor: int = 200
    learning_rate: float = 8e-6
    ode_hidden: int = 7
    ode_layer_width: int = 200
    ode_tanh_reg: bool = False
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the simple_reaction dataset"""

    hidden_size: int = 392
    num_hidden_layers: int = 3
    learning_rate: float = 5e-6
    activation: nn.Module = nn.GELU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the simple_reaction dataset"""

    latent_features: int = 10
    degree: int = 1
    learning_rate: float = 0.0005
    layers_factor: int = 159
    activation: nn.Module = nn.LeakyReLU()
