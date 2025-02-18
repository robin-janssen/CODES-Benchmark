from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the osu2008 dataset"""

    branch_hidden_layers: int = 5
    trunk_hidden_layers: int = 4
    hidden_size: int = 100
    output_factor: int = 50
    learning_rate: float = 1e-5
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the osu2008 dataset"""

    latent_features: int = 10
    layers_factor: int = 49
    learning_rate: float = 0.002
    ode_tanh_reg: bool = True
    activation: nn.Module = nn.Tanh()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the osu2008 dataset"""

    hidden_size: int = 478
    num_hidden_layers: int = 1
    learning_rate: float = 2e-5
    activation: nn.Module = nn.Softplus()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the osu2008 dataset"""

    latent_features: int = 10
    degree: int = 1
    learning_rate: float = 2e-4
    layers_factor: int = 88
    activation: nn.Module = nn.GELU()
