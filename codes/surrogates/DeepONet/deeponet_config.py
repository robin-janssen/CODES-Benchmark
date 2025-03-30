from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetBaseConfig:
    """Standard model config for MultiONet"""

    masses: list[float] | None = None  # field(default_factory=lambda: osu_masses)
    trunk_input_size: int = 1
    hidden_size: int = 100
    branch_hidden_layers: int = 5
    trunk_hidden_layers: int = 5
    output_factor: int = 10
    learning_rate: float = 3e-4
    schedule: bool = False
    regularization_factor: float = 0.0  # 0.012
    massloss_factor: float = 0.0  # 0.012
    activation: nn.Module = nn.ReLU()
