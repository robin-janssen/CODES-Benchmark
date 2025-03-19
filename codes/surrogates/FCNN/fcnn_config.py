from dataclasses import dataclass

import torch.nn as nn

# from typing import Optional


@dataclass
class FCNNBaseConfig:
    """Standard model config for FCNN"""

    hidden_size: int = 150
    num_hidden_layers: int = 5
    learning_rate: float = 1e-4
    schedule: bool = False
    regularization_factor: float = 0.012
    activation: nn.Module = nn.ReLU()
