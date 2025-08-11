from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the primordial_parametric dataset"""

    # primordial_parametric_final_multionet, trial 18
    scheduler: str = "poly"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.MSELoss()
    activation: nn.Module = nn.PReLU()
    branch_hidden_layers: int = 5
    hidden_size: int = 50
    learning_rate: float = 0.0018
    output_factor: int = 74
    params_branch: bool = True
    poly_power: float = 0.64
    regularization_factor: float = 0.00114
    trunk_hidden_layers: bool = True


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the primordial_parametric dataset"""

    # primordial_parametric_final_latentneuralode, trial 234
    scheduler: str = "cosine"
    optimizer: str = "SGD"
    loss_function: nn.Module = nn.MSELoss()
    activation: nn.Module = nn.SiLU()
    coder_layers: bool = True
    coder_width: int = 360
    encode_params: bool = False
    eta_min: float = 0.00191
    latent_features: int = 10
    learning_rate: float = 2.04e-05
    momentum: float = 0.823
    ode_layers: int = 8
    ode_tanh_reg: bool = True
    ode_width: int = 220
    regularization_factor: float = 1.87e-05


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the primordial_parametric dataset"""

    # primordial_parametric_final_fullyconnected, trial 1
    scheduler: str = "poly"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.SmoothL1Loss()
    activation: nn.Module = nn.Mish()
    beta: float = 6.69
    hidden_size: int = 470
    learning_rate: float = 0.00129
    num_hidden_layers: int = 3
    poly_power: float = 0.899
    regularization_factor: float = 0.0144


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the primordial_parametric dataset"""

    # primordial_parametric_final_latentpoly, trial 16
    scheduler: str = "poly"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.MSELoss()
    activation: nn.Module = nn.ELU()
    coder_layers: bool = True
    coder_width: int = 700
    coeff_network: bool = False
    degree: int = 3
    latent_features: int = 7
    learning_rate: float = 1.33e-05
    poly_power: float = 0.804
    regularization_factor: float = 0.0174


# @dataclass
# class MultiONetConfig:
#     """Model config for MultiONet for the simple_ode dataset"""

#     # primordialparams, trial 56

#     branch_hidden_layers: int = 2
#     trunk_hidden_layers: int = 7
#     hidden_size: int = 164
#     output_factor: int = 63
#     learning_rate: float = 4e-5
#     activation: nn.Module = nn.GELU()
#     params_branch: bool = True


# @dataclass
# class LatentNeuralODEConfig:
#     """Model config for LatentNeuralODE for the simple_ode dataset"""

#     # primordialparams, trial 75

#     latent_features: int = 5
#     coder_layers: int = 4
#     coder_width: int = 268
#     learning_rate: float = 4e-5
#     ode_layers: int = 9
#     ode_width: int = 125
#     ode_tanh_reg: bool = False
#     activation: nn.Module = nn.Tanh()
#     model_version: str = "v2"
#     encode_params: bool = False


# @dataclass
# class FullyConnectedConfig:
#     """Model config for FullyConnected for the simple_ode dataset"""

#     # primordialparams, trial 40

#     hidden_size: int = 629
#     num_hidden_layers: int = 1
#     learning_rate: float = 6e-5
#     activation: nn.Module = nn.LeakyReLU()


# @dataclass
# class LatentPolyConfig:
#     """Model config for LatentPoly for the simple_ode dataset"""

#     # primordialparams, trial 32

#     latent_features: int = 10
#     degree: int = 7
#     learning_rate: float = 5e-4
#     coder_layers: int = 2
#     coder_width: int = 267
#     activation: nn.Module = nn.LeakyReLU()
#     coeff_network: bool = False
