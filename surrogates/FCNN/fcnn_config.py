from dataclasses import dataclass

# from typing import Optional


@dataclass
class FCNNBaseConfig:
    """Standard model config for FCNN"""

    hidden_size: int = 150
    num_hidden_layers: int = 5
    learning_rate: float = 1e-4
    schedule: bool = False
    regularization_factor: float = 0.012


@dataclass
class OConfig:
    """Only for backward compatibility with old models"""

    hidden_size: int = 150
    num_hidden_layers: int = 5
    learning_rate: float = 1e-4
    schedule: bool = False
    regularization_factor: float = 0.012
