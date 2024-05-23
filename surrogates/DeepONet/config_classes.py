from dataclasses import dataclass, field
from typing import Optional

# from optuna import Trial

from data.osu_data.osu_chemicals import osu_masses


@dataclass
class PChemicalTrainConfig:
    """Dataclass for the configuration of a multionet model for the priestley chemicals dataset."""

    masses: Optional[list[float]] = None
    branch_input_size: int = 216
    trunk_input_size: int = 1
    hidden_size: int = 767
    branch_hidden_layers: int = 4
    trunk_hidden_layers: int = 6
    output_neurons: int = 4320
    N_outputs: int = 216
    num_epochs: int = 20
    learning_rate: float = 9.396e-06
    schedule: bool = False
    N_sensors: int = 216
    N_timesteps: int = 128
    architecture: str = "both"
    pretrained_model_path: Optional[str] = None
    device: str = "cpu"
    use_streamlit: bool = False
    # optuna_trial: Trial | None = None
    regularization_factor: float = 0.0
    massloss_factor: float = 0.0
    batch_size: int = 256


@dataclass
class BChemicalTrainConfig:
    """Dataclass for the configuration of a multionet model for the branca chemicals dataset."""

    masses: Optional[list[float]] = None
    branch_input_size: int = 10
    trunk_input_size: int = 1
    hidden_size: int = 250
    branch_hidden_layers: int = 4
    trunk_hidden_layers: int = 4
    output_neurons: int = 360
    N_outputs: int = 10
    num_epochs: int = 500
    learning_rate: float = 3e-5
    schedule: bool = False
    N_sensors: int = 10
    N_timesteps: int = 16
    architecture: str = "both"
    pretrained_model_path: Optional[str] = None
    device: str = "cuda:0"
    use_streamlit: bool = False
    # optuna_trial: Trial | None = None
    regularization_factor: float = 0.0
    massloss_factor: float = 0.0
    batch_size: int = 512


@dataclass
class OChemicalTrainConfig:
    """Dataclass for the configuration of a multionet model for the osu chemicals dataset."""

    masses: Optional[list[float]] = field(default_factory=lambda: osu_masses)
    branch_input_size: int = 29
    trunk_input_size: int = 1
    hidden_size: int = 100
    branch_hidden_layers: int = 5
    trunk_hidden_layers: int = 5
    output_neurons: int = 290
    N_outputs: int = 29
    num_epochs: int = 10
    learning_rate: float = 3e-4
    schedule: bool = False
    N_sensors: int = 29
    N_timesteps: int = 100
    architecture: str = "both"
    pretrained_model_path: Optional[str] = None
    device: str = "mps"
    use_streamlit: bool = False
    # optuna_trial: Trial | None = None
    regularization_factor: float = 0.012
    massloss_factor: float = 0.012
    batch_size: int = 128


@dataclass
class SpectraTrainConfig:
    """Dataclass for the configuration of a multionet model for the spectral dataset."""

    branch_input_size: int = 92
    trunk_input_size: int = 2
    hidden_size: int = 100
    branch_hidden_layers: int = 3
    trunk_hidden_layers: int = 3
    output_neurons: int = 100
    N_outputs: int = 1
    num_epochs: int = 10
    learning_rate: float = 1e-4
    schedule: bool = False
    N_sensors: int = 92
    N_timesteps: int = 11
    pretrained_model_path: Optional[str] = None
    device: str = "mps"
    use_streamlit: bool | None = None
    # optuna_trial: Trial | None = None
    regularization_factor: float = 0.0
    batch_size: int = 256
