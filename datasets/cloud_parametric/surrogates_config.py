from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the cloud_parametric dataset"""

    # cloud_parametric_final_multionet, trial 27
    scheduler: str = "poly"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.MSELoss()
    activation: nn.Module = nn.ELU()
    branch_hidden_layers: int = 4
    hidden_size: int = 100
    learning_rate: float = 0.00184
    output_factor: int = 40
    params_branch: bool = True
    poly_power: float = 1.52
    regularization_factor: float = 0.0171
    trunk_hidden_layers: int = 4


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the cloud_parametric dataset"""

    # cloud_parametric_final_latentneuralode, trial 299
    scheduler: str = "schedulefree"
    optimizer: str = "SGD"
    loss_function: nn.Module = nn.SmoothL1Loss()
    activation: nn.Module = nn.GELU()
    beta: float = 9.61
    coder_layers: int = 2
    coder_width: int = 180
    encode_params: bool = False
    latent_features: int = 10
    learning_rate: float = 0.00693
    momentum: float = 0.613
    ode_layers: int = 3
    ode_tanh_reg: bool = False
    ode_width: int = 220
    regularization_factor: float = 0.000134


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the cloud_parametric dataset"""

    # cloud_parametric_final_fullyconnected, trial 61
    scheduler: str = "poly"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.SmoothL1Loss()
    activation: nn.Module = nn.ELU()
    beta: float = 0.299
    hidden_size: int = 290
    learning_rate: float = 0.00331
    num_hidden_layers: int = 5
    poly_power: float = 1.88
    regularization_factor: float = 0.113


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the cloud_parametric dataset"""

    # cloud_parametric_final_latentpoly, trial 13
    scheduler: str = "poly"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.SmoothL1Loss()
    activation: nn.Module = nn.ReLU()
    beta: float = 2.88
    coder_layers: bool = True
    coder_width: int = 170
    coeff_network: bool = False
    degree: int = 9
    latent_features: int = 10
    learning_rate: float = 0.00551
    poly_power: float = 1.65
    regularization_factor: float = 0.00629


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

#     # cloudparams, trial 40
#     latent_features: int = 3
#     coder_layers: int = 3
#     coder_width: int = 377
#     learning_rate: float = 3e-4
#     ode_layers: int = 5
#     ode_width: int = 167
#     regularization_factor: float = 0.000127
#     encode_params: bool = False
#     optimizer: str = "sgd"
#     momentum: float = 0.226
#     scheduler: str = "cosine"
#     eta_min: float = 0.0222
#     ode_tanh_reg: bool = False
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
