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
    layers_factor: int = 200
    learning_rate: float = 0.0004
    ode_hidden: int = 9
    ode_layer_width: int = 58
    ode_tanh_reg: bool = True
    activation: nn.Module = nn.LeakyReLU()


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
    layers_factor: int = 64
    activation: nn.Module = nn.LeakyReLU()
