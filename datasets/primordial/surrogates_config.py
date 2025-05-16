from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the simple_ode dataset"""

    branch_hidden_layers: int = 8
    trunk_hidden_layers: int = 8
    hidden_size: int = 130
    output_factor: int = 98
    learning_rate: float = 2e-4
    activation: nn.Module = nn.Tanh()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the simple_ode dataset"""

    latent_features: int = 10
    coder_layers: int = 4
    coder_width: int = 350
    learning_rate: float = 2e-4
    ode_layers: int = 10
    ode_width: int = 157
    ode_tanh_reg: bool = True
    activation: nn.Module = nn.Softplus()
    model_version: str = "v2"


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the simple_ode dataset"""

    hidden_size: int = 453
    num_hidden_layers: int = 1
    learning_rate: float = 5e-4
    activation: nn.Module = nn.ReLU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the simple_ode dataset"""

    latent_features: int = 9
    degree: int = 1
    learning_rate: float = 2e-4
    coder_layers: int = 2
    coder_width: int = 264
    activation: nn.Module = nn.LeakyReLU()
