from dataclasses import dataclass

from torch import nn


@dataclass
class FCNNBaseConfig:
    """
    Configuration for the Fully Connected Neural Network (FCNN) surrogate model.

    This config defines the structure and training settings for a standard MLP-style
    surrogate model with fully connected layers.

    Attributes:
        hidden_size (int): Number of neurons per hidden layer.
        num_hidden_layers (int): Number of hidden layers in the network.
        learning_rate (float): Learning rate for the optimizer.
        schedule (bool): Whether to use learning rate scheduling.
        regularization_factor (float): L2 regularization coefficient.
        activation (nn.Module): Activation function applied between layers.
    """

    hidden_size: int = 150
    num_hidden_layers: int = 5
    learning_rate: float = 1e-4
    schedule: bool = False
    regularization_factor: float = 0.012
    activation: nn.Module = nn.ReLU()
