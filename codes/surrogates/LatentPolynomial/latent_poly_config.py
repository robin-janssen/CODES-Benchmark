from dataclasses import dataclass

from torch import nn


@dataclass
class LatentPolynomialBaseConfig:
    """
    Standard model config for LatentPolynomial with versioning.

    Attributes:
        model_version (str): "v1" for the old fixed 4–2–1 structure; "v2" for the new FCNN design.
        latent_features (int): Dimension of the latent space.
        degree (int): Degree of the learnable polynomial.
        coder_hidden (int): (v1 only) Base hidden size for fixed structure.
        layers_factor (int): (v1 only) Factor multiplied with coder_hidden to determine layer widths.
        coder_layers (int): (v2 only) Number of hidden layers in the encoder/decoder.
        coder_width (int): (v2 only) Number of neurons in every hidden layer in the encoder/decoder.
        activation (nn.Module): Activation function.
        learning_rate (float): Learning rate for training.
    """

    model_version: str = "v1"  # Default to new architecture.
    latent_features: int = 5
    degree: int = 2
    coder_hidden: int = 4  # Used in v1 only.
    layers_factor: int = 8  # Used in v1 only.
    coder_layers: int = 3  # Used in v2 only.
    coder_width: int = 32  # Used in v2 only.
    activation: nn.Module = nn.ReLU()
    learning_rate: float = 1e-3
