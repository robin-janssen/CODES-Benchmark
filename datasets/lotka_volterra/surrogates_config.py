from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the osu2008 dataset"""

    branch_hidden_layers: int = 6
    trunk_hidden_layers: int = 9
    hidden_size: int = 313
    output_factor: int = 72
    learning_rate: float = 0.0003
    activation: nn.Module = nn.ReLU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the osu2008 dataset"""

    latent_features: int = 5
    layers_factor: int = 61
    learning_rate: float = 0.002
    ode_activation: nn.Module = nn.Tanh()
    ode_tanh_reg: bool = True
    coder_activation: nn.Module = nn.Tanh()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the osu2008 dataset"""

    hidden_size: int = 320
    num_hidden_layers: int = 1  # Can this really be 1?
    learning_rate: float = 2e-3
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the osu2008 dataset"""

    latent_features: int = 6
    degree: int = 1
    learning_rate: float = 2e-4
    layers_factor: int = 78
    coder_activation: nn.Module = nn.LeakyReLU()
