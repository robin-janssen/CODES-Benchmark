from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the simple_ode dataset"""

    # Naive configuration
    loss_function: nn.Module = nn.MSELoss()
    optimizer: str = "adamw"
    scheduler: str = "schedulefree"
    branch_hidden_layers: int = 5
    hidden_size: int = 256
    output_factor: int = 200
    trunk_hidden_layers: int = 5
    learning_rate: float = 1e-04
    regularization_factor: float = 1e-04
    activation: nn.Module = nn.ReLU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the simple_ode dataset"""

    # Naive configuration
    loss_function: nn.Module = nn.MSELoss()
    optimizer: str = "adamw"
    scheduler: str = "schedulefree"
    ode_tanh_reg: bool = False
    latent_features: int = 10
    coder_layers: int = 3
    coder_width: int = 256
    ode_layers: int = 3
    ode_width: int = 256
    learning_rate: float = 1e-04
    regularization_factor: float = 1e-04
    activation: nn.Module = nn.ReLU()


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the simple_ode dataset"""

    # Naive configuration
    loss_function: nn.Module = nn.MSELoss()
    optimizer: str = "adamw"
    scheduler: str = "schedulefree"
    # other params from primordial_tuning, trial 63
    hidden_size: int = 256
    num_hidden_layers: int = 5
    learning_rate: float = 1e-04
    regularization_factor: float = 1e-04
    activation: nn.Module = nn.ReLU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the simple_ode dataset"""

    # Naive configuration
    loss_function: nn.Module = nn.MSELoss()
    optimizer: str = "adamw"
    scheduler: str = "schedulefree"
    degree: int = 3
    latent_features: int = 10
    coder_layers: int = 3
    coder_width: int = 256
    learning_rate: float = 1e-04
    regularization_factor: float = 1e-04
    activation: nn.Module = nn.ReLU()


# @dataclass
# class MultiONetConfig:
#     """Model config for MultiONet for the simple_ode dataset"""

#     # primordial_tuning, trial 18
#     scheduler: str = "poly"
#     optimizer: str = "AdamW"
#     loss_function: nn.Module = nn.SmoothL1Loss()
#     poly_power: float = 0.725
#     beta: float = 0.568
#     branch_hidden_layers: int = 5
#     hidden_size: int = 560
#     output_factor: int = 293
#     trunk_hidden_layers: int = 5
#     learning_rate: float = 5.4e-04
#     regularization_factor: float = 1.8e-02
#     activation: nn.Module = nn.GELU()


# @dataclass
# class LatentNeuralODEConfig:
#     """Model config for LatentNeuralODE for the simple_ode dataset"""

#     # primordial_tuning, trial 186
#     scheduler: str = "poly"
#     optimizer: str = "sgd"
#     loss_function: nn.Module = nn.SmoothL1Loss()
#     poly_power: float = 0.948
#     momentum: float = 0.702
#     beta: float = 0.684
#     latent_features: int = 10
#     coder_layers: int = 1
#     coder_width: int = 490
#     ode_layers: int = 3
#     ode_width: int = 50  # went up from 20 - very cheap increase in expressivity
#     learning_rate: float = 1.99e-4
#     regularization_factor: float = 7.66e-02
#     activation: nn.Module = nn.Mish()


# @dataclass
# class FullyConnectedConfig:
#     """Model config for FullyConnected for the simple_ode dataset"""

#     # primordial_tuning, trial 63
#     scheduler: str = "poly"
#     optimizer: str = "AdamW"
#     loss_function: nn.Module = nn.SmoothL1Loss()
#     poly_power: float = 1.691
#     beta: float = 2.611
#     hidden_size: int = 470
#     num_hidden_layers: int = 4
#     learning_rate: float = 2.3e-03
#     regularization_factor: float = 0.309
#     activation: nn.Module = nn.LeakyReLU()


# @dataclass
# class LatentPolyConfig:
#     """Model config for LatentPoly for the simple_ode dataset"""

#     # primordial_tuning, trial 176
#     scheduler: str = "poly"
#     optimizer: str = "AdamW"
#     loss_function: nn.Module = nn.SmoothL1Loss()
#     poly_power: float = 0.845
#     beta: float = 2.462
#     degree: int = 4
#     latent_features: int = 8
#     coder_layers: int = 2
#     coder_width: int = 150
#     learning_rate: float = 9.36e-04
#     regularization_factor: float = 4.9e-04
#     activation: nn.Module = nn.GELU()
