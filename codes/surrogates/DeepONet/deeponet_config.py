from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetBaseConfig:
    """
    Configuration for the MultiONet surrogate model.

    MultiONet is a physics-inspired architecture that splits the network into a
    trunk and multiple branch networks, designed to represent operator-based systems.

    Attributes:
        masses (list[float] | None): Optional list of particle masses (if used).
        trunk_input_size (int): Size of the input to the trunk network.
        hidden_size (int): Number of neurons per hidden layer in both branches and trunk.
        branch_hidden_layers (int): Number of hidden layers in each branch network.
        trunk_hidden_layers (int): Number of hidden layers in the trunk network.
        output_factor (int): Multiplier for the output dimension (neurons = output_factor * num_quantities).
        learning_rate (float): Learning rate for the optimizer.
        schedule (bool): Whether to apply a learning rate scheduler.
        regularization_factor (float): L2 regularization coefficient.
        massloss_factor (float): Additional weight for a mass conservation loss term.
        activation (nn.Module): Activation function used in all layers.
        params_branch (bool): Flag to indicate whether parameters (if present) are passed to the branch or trunk net.
    """

    masses: list[float] | None = None
    trunk_input_size: int = 1
    hidden_size: int = 100
    branch_hidden_layers: int = 5
    trunk_hidden_layers: int = 5
    output_factor: int = 10
    learning_rate: float = 3e-4
    schedule: bool = False
    regularization_factor: float = 0.0
    massloss_factor: float = 0.0
    activation: nn.Module = nn.ReLU()
    params_branch: False
