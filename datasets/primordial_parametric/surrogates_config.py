from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the simple_ode dataset"""

    # primordialparams, trial 56

    branch_hidden_layers: int = 2
    trunk_hidden_layers: int = 7
    hidden_size: int = 164
    output_factor: int = 63
    learning_rate: float = 4e-5
    activation: nn.Module = nn.GELU()
    params_branch: bool = True


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the simple_ode dataset"""

    # primordialparams, trial 75

    latent_features: int = 5
    coder_layers: int = 4
    coder_width: int = 268
    learning_rate: float = 4e-5
    ode_layers: int = 9
    ode_width: int = 125
    ode_tanh_reg: bool = False
    activation: nn.Module = nn.Tanh()
    model_version: str = "v2"
    encode_params: bool = False


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the simple_ode dataset"""

    # primordialparams, trial 40

    hidden_size: int = 629
    num_hidden_layers: int = 1
    learning_rate: float = 6e-5
    activation: nn.Module = nn.LeakyReLU()


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the simple_ode dataset"""

    # primordialparams, trial 32

    latent_features: int = 10
    degree: int = 7
    learning_rate: float = 5e-4
    coder_layers: int = 2
    coder_width: int = 267
    activation: nn.Module = nn.LeakyReLU()
    coeff_network: bool = False
