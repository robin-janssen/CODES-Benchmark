from dataclasses import dataclass

from torch import nn

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
