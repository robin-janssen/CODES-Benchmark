from dataclasses import dataclass

from codes.surrogates.AbstractSurrogate import AbstractSurrogateBaseConfig


@dataclass
class LatentPolynomialBaseConfig(AbstractSurrogateBaseConfig):
    """
    Standard model config for LatentPolynomial with versioning.

    Attributes:
        model_version (str): "v1" for the old fixed 4-2-1 structure; "v2" for the new FCNN design.
        latent_features (int): Dimension of the latent space.
        degree (int): Degree of the learnable polynomial.
        coder_hidden (int): (v1 only) Base hidden size for fixed structure.
        layers_factor (int): (v1 only) Factor multiplied with coder_hidden to determine layer widths.
        coder_layers (int): (v2 only) Number of hidden layers in the encoder/decoder.
        coder_width (int): (v2 only) Number of neurons in every hidden layer in the encoder/decoder.
        coeff_network (bool): Whether to use a coefficient network for polynomial coefficients.
        coeff_width (int): Width of the coefficient network (if used).
        coeff_layers (int): Number of layers in the coefficient network (if used).
    """

    model_version: str = "v2"  # Default to new architecture.
    latent_features: int = 5
    degree: int = 2
    coder_hidden: int = 4  # Used in v1 only.
    layers_factor: int = 8  # Used in v1 only.
    coder_layers: int = 3  # Used in v2 only.
    coder_width: int = 32  # Used in v2 only.
    coeff_network: bool = False
    coeff_width: int = 32  # Used only if coeff_network == True
    coeff_layers: int = 4  # Used only if coeff_network == True
