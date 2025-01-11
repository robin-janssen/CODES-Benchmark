from dataclasses import dataclass  # , field

from torch import nn


@dataclass
class LatentNeuralODEBaseConfig:
    """Standard model config for LatentNeuralODE"""

    latent_features: int = 5
    # coder_layers: list[int] = field(default_factory=lambda: [32, 16, 8])
    activation: nn.Module = nn.ReLU()
    ode_hidden: int = 4
    ode_layer_width: int = 64
    ode_tanh_reg: bool = True
    layers_factor: int = 8
    rtol: float = 1e-5
    atol: float = 1e-5
    t_steps = 100
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-5
