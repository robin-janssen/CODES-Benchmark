from dataclasses import dataclass

# from typing import Optional


@dataclass
class OConfig:
    """Dataclass for the configuration of a fully connected neural network for the osu chemicals dataset."""

    hidden_size: int = 150
    num_hidden_layers: int = 5
    learning_rate: float = 1e-4
    schedule: bool = False
    regularization_factor: float = 0.012
