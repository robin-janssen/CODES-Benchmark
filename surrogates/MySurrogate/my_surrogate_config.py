from dataclasses import dataclass

from torch.nn import ReLU, Module


@dataclass
class MySurrogateConfig:
    """Model config for MySurrogate for the osu2008 dataset"""

    hidden_layers: int = 2
    layer_width: int = 128
    activation: Module = ReLU()
    learning_rate: float = 0.001