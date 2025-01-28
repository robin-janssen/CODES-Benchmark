from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the osu2008 dataset"""

    branch_hidden_layers: int = 2
    trunk_hidden_layers: int = 7
    hidden_size: int = 453
    output_factor: int = 58
    learning_rate: float = 4.5e-05
    activation: nn.Module = nn.ReLU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the osu2008 dataset"""

    latent_features: int = 10
    layers_factor: int = 88
    learning_rate: float = 0.0025
    ode_hidden: int = 3
    ode_layer_width: int = 402
    ode_tanh_reg: bool = True
    activation: nn.Module = nn.GELU()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the osu2008 dataset"""

    hidden_size: int = 406
    num_hidden_layers: int = 4
    learning_rate: float = 1.0e-05
    activation: nn.Module = nn.GELU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the osu2008 dataset"""

    latent_features: int = 9
    degree: int = 1
    learning_rate: float = 0.0035  # 0.001
    layers_factor: int = 96
    activation: nn.Module = nn.Tanh()
