from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the simple_ode dataset"""

    loss_function: nn.Module = nn.SmoothL1Loss()
    optimizer: str = "adamw"
    scheduler: str = "schedulefree"
    # other params from cloud_tuning_fine, trial 81:
    beta: float = 1.78
    branch_hidden_layers: int = 6
    hidden_size = 170
    output_factor: int = 80
    trunk_hidden_layers: int = 7
    learning_rate: float = 8.3e-4
    regularization_factor: float = 9.6e-06
    activation: nn.Module = nn.Tanh()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the simple_ode dataset"""

    loss_function: nn.Module = nn.MSELoss()
    optimizer: str = "adamw"
    scheduler: str = "schedulefree"
    ode_tanh_reg: bool = False
    # other params from cloud_tuning_fine, trial 35:
    latent_features: int = 14
    coder_layers: int = 4
    coder_width: int = 70
    ode_width: int = 30
    ode_layers: int = 2
    learning_rate: float = 7.1e-03
    regularization_factor: float = 3.4e-03
    activation: nn.Module = nn.Tanh()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the simple_ode dataset"""

    loss_function: nn.Module = nn.SmoothL1Loss()
    optimizer: str = "adamw"
    scheduler: str = "schedulefree"
    # other params from cloud_tuning_fine, trial 77
    beta = 0.817
    hidden_size: int = 380
    num_hidden_layers: int = 2
    learning_rate: float = 6.8e-03
    regularization_factor: float = 8.7e-06
    activation: nn.Module = nn.ReLU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the simple_ode dataset"""

    loss_function: nn.Module = nn.MSELoss()
    optimizer: str = "adamw"
    scheduler: str = "schedulefree"
    # other params from cloud_tuning_fine, trial 89:
    degree: int = 3
    latent_features: int = 9
    coder_layers: int = 3
    coder_width: int = 210
    learning_rate: float = 2.2e-5
    regularization_factor: float = 1.7e-03


# @dataclass
# class MultiONetConfig:
#     """Model config for MultiONet for the simple_ode dataset"""

#     # cloud, trial 69
#     branch_hidden_layers: int = 1
#     trunk_hidden_layers: int = 9
#     hidden_size: int = 225
#     output_factor: int = 63
#     learning_rate: float = 4e-5  # optimal for ~4000 epochs
#     activation: nn.Module = nn.Tanh()


# @dataclass
# class LatentNeuralODEConfig:
#     """Model config for LatentNeuralODE for the simple_ode dataset"""

#     # cloud, trial 63
#     latent_features: int = 6
#     coder_layers: int = 2
#     coder_width: int = 103
#     learning_rate: float = 3e-4
#     ode_layers: int = 4
#     ode_width: int = 197
#     ode_tanh_reg: bool = True
#     activation: nn.Module = nn.SiLU()
#     model_version: str = "v2"


# @dataclass
# class FullyConnectedConfig:
#     """Model config for FullyConnected for the simple_ode dataset"""

#     # cloud, trial 44
#     hidden_size: int = 261
#     num_hidden_layers: int = 1
#     learning_rate: float = 1e-4
#     activation: nn.Module = nn.LeakyReLU()


# @dataclass
# class LatentPolyConfig:
#     """Model config for LatentPoly for the simple_ode dataset"""

#     # cloud, trial 92
#     latent_features: int = 9
#     degree: int = 5
#     learning_rate: float = 3e-4
#     coder_layers: int = 1
#     coder_width: int = 86
#     activation: nn.Module = nn.Mish()
