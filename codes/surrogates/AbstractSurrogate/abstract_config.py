from dataclasses import dataclass

from torch import nn


@dataclass
class AbstractSurrogateBaseConfig:
    """
    Base configuration for the AbstractSurrogate model.

    This class defines shared attributes and methods for surrogate models.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        regularization_factor (float): Regularization coefficient, applied as weight decay.
        scheduler (str): Type of learning rate scheduler to use.
            - "schedulefree": Use schedulefree optimizer.
            - "cosine": Use cosine annealing scheduler.
            - "poly": Use polynomial decay scheduler.
        poly_power (float): Power for polynomial decay scheduler (used only if scheduler == "poly").
        eta_min (float): Multiplier for minimum learning rate for cosine annealing scheduler (used only if scheduler == "cosine").
        activation (nn.Module): Activation function used in the model.
        momentum (float): Momentum factor for the optimizer (used only if optimizer == "sgd").
    """

    learning_rate: float = 3e-4
    regularization_factor: float = 0.0
    scheduler: str = "schedulefree"
    poly_power: float = 0.9  # Used only if scheduler == "poly"
    eta_min: float = 1e-1  # Used only if scheduler == "cosine"
    activation: nn.Module = nn.ReLU()
    momentum: float = 0.0  # Used only if optimizer == "sgd"
