from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the simple_ode dataset"""

    # lvparams3 run, trial 22

    branch_hidden_layers: int = 9
    trunk_hidden_layers: int = 6
    hidden_size: int = 695
    output_factor: int = 371
    learning_rate: float = 6.7e-4
    params_branch: bool = True
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the simple_ode dataset"""

    # lvparams3 run, trial 12

    latent_features: int = 10
    coder_layers: int = 4
    coder_width: int = 188
    learning_rate: float = 1.8e-3
    activation: nn.Module = nn.LeakyReLU()
    ode_tanh_reg: bool = True
    ode_width: int = 409
    ode_layers: int = 9
    encode_params: bool = True
    model_version: str = "v2"


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the simple_ode dataset"""

    # lvparams run, trial 61

    hidden_size: int = 970
    num_hidden_layers: int = 3
    learning_rate: float = 7.6e-4
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the simple_ode dataset"""

    # lvparams5 run, trial 33

    activation: nn.Module = nn.GELU()
    degree: int = 2
    latent_features: int = 5
    coder_layers: int = 5
    coder_width: int = 480
    learning_rate: float = 0.002
    coeff_network: bool = False
