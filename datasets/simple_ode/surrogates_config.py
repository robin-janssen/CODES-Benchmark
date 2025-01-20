from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the osu2008 dataset"""

    branch_hidden_layers: int = 5
    trunk_hidden_layers: int = 5
    hidden_size: int = 100
    output_factor: int = 10
    learning_rate: float = 0.0005
    activation: nn.Module = nn.LeakyReLU()
