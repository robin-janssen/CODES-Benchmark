from dataclasses import dataclass
from typing import Optional


@dataclass
class OConfig:
    """Dataclass for the configuration of a fully connected neural network for the osu chemicals dataset."""

    input_size: int = 30  # 29 chemicals + 1 time input
    hidden_size: int = 150
    num_hidden_layers: int = 5
    output_size: int = 29
    num_epochs: int = 250
    learning_rate: float = 1e-4
    schedule: bool = False
    N_timesteps: int = 100
    pretrained_model_path: Optional[str] = None
    device: str = "mps"
    batch_size: int = 1024
    regularization_factor: float = 0.012
