from dataclasses import dataclass

from AbstractSurrogate import AbstractSurrogateBaseConfig


@dataclass
class FCNNBaseConfig(AbstractSurrogateBaseConfig):
    """
    Configuration for the Fully Connected Neural Network (FCNN) surrogate model.

    This config defines the structure and training settings for a standard MLP-style
    surrogate model with fully connected layers.

    Attributes:
        hidden_size (int): Number of neurons per hidden layer.
        num_hidden_layers (int): Number of hidden layers in the network.
    """

    hidden_size: int = 150
    num_hidden_layers: int = 5
