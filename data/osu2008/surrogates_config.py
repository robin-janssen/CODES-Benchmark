from dataclasses import dataclass


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the osu2008 dataset"""

    branch_hidden_layers: int = 5
    trunk_hidden_layers: int = 5


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the osu2008 dataset"""

    latent_features: int = 5
    coder_hidden: int = 4


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the osu2008 dataset"""

    hidden_size: int = 150
    num_hidden_layers: int = 5


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the osu2008 dataset"""

    latent_features: int = 5
    degree: int = 2
