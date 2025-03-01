from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the lotka_volterra dataset"""

    branch_hidden_layers: int = 8
    trunk_hidden_layers: int = 3
    hidden_size: int = 267
    output_factor: int = 48
    learning_rate: float = 2.8e-05  # 0.0001
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the lotka_volterra dataset"""

    latent_features: int = 8
    layers_factor: int = 83
    learning_rate: float = 0.0023
    ode_hidden: int = 4
    ode_layer_width: int = 116
    ode_tanh_reg: bool = False
    activation: nn.Module = nn.GELU()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the lotka_volterra dataset"""

    hidden_size: int = 222
    num_hidden_layers: int = 2
    learning_rate: float = 1.2e-05  # 0.0001
    activation: nn.Module = nn.Tanh()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the lotka_volterra dataset"""

    latent_features: int = 2
    degree: int = 1
    learning_rate: float = 0.00025  # 0.001
    layers_factor: int = 61
    activation: nn.Module = nn.Tanh()
