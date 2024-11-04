from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the osu2008 dataset"""

    branch_hidden_layers: int = 6
    trunk_hidden_layers: int = 5
    hidden_size: int = 200
    output_factor: int = 38
    learning_rate: float = 0.0003
    activation: nn.Module = nn.GELU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the osu2008 dataset"""

    latent_features: int = 10
    layers_factor: int = 90
    learning_rate: float = 0.001
    ode_activation: nn.Module = nn.GELU()
    ode_tanh_reg: bool = True
    coder_activation: nn.Module = nn.Tanh()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the osu2008 dataset"""

    hidden_size: int = 175
    num_hidden_layers: int = 1  # Can this really be 1?
    learning_rate: float = 1e-4
    activation: nn.Module = nn.ReLU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the osu2008 dataset"""

    latent_features: int = 13
    degree: int = 1
    learning_rate: float = 0.0003
    layers_factor: int = 58
    coder_activation: nn.Module = nn.GELU()
