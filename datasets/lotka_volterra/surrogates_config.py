from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the osu2008 dataset"""

    branch_hidden_layers: int = 5
    trunk_hidden_layers: int = 9
    hidden_size: int = 275
    output_factor: int = 98
    learning_rate: float = 0.00004  # 0.0001
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the osu2008 dataset"""

    latent_features: int = 5
    layers_factor: int = 85
    learning_rate: float = 0.0005
    ode_hidden: int = 3
    ode_layer_width: int = 412
    ode_tanh_reg: bool = True
    activation: nn.Module = nn.Tanh()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the osu2008 dataset"""

    hidden_size: int = 486
    num_hidden_layers: int = 2
    learning_rate: float = 0.00008
    activation: nn.Module = nn.Tanh()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the osu2008 dataset"""

    latent_features: int = 8
    degree: int = 7
    learning_rate: float = 0.0004  # 0.001
    layers_factor: int = 84
    activation: nn.Module = nn.ReLU()


# @dataclass
# class MultiONetConfig:
#     """Model config for MultiONet for the osu2008 dataset"""

#     branch_hidden_layers: int = 4
#     trunk_hidden_layers: int = 7
#     hidden_size: int = 150
#     output_factor: int = 40
#     learning_rate: float = 0.0005
#     activation: nn.Module = nn.LeakyReLU()


# @dataclass
# class LatentNeuralODEConfig:
#     """Model config for LatentNeuralODE for the osu2008 dataset"""

#     latent_features: int = 9
#     layers_factor: int = 46
#     learning_rate: float = 0.005
#     ode_tanh_reg: bool = False
#     activation: nn.Module = nn.ReLU()


# @dataclass
# class FullyConnectedConfig:
#     """Model config for FullyConnected for the osu2008 dataset"""

#     hidden_size: int = 400
#     num_hidden_layers: int = 2
#     learning_rate: float = 1.5e-5
#     activation: nn.Module = nn.Tanh()


# @dataclass
# class LatentPolyConfig:
#     """Model config for LatentPoly for the osu2008 dataset"""

#     latent_features: int = 5
#     degree: int = 6
#     learning_rate: float = 0.002
#     layers_factor: int = 50
#     activation: nn.Module = nn.ReLU()
