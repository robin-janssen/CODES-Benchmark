import os
import yaml
import torch
from typing import Dict


def check_surrogate(surrogate: str, conf: Dict) -> None:
    """
    Check whether the required models for the benchmark are present in the expected directories.

    Args:
        surrogate (str): The name of the surrogate model to check.
        conf (dict): The configuration dictionary.

    Raises:
        FileNotFoundError: If any required models are missing.
    """
    training_id = conf["training_ID"]
    base_dir = os.getcwd()
    base_dir = os.path.join(base_dir, "trained", surrogate, training_id)

    required_models = get_required_models_list(surrogate, conf)

    for model_name in required_models:
        model_path = os.path.join(base_dir, f"{model_name}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Required model {model_name} for surrogate {surrogate} not found in {base_dir}"
            )

    print(f"All required models for surrogate {surrogate} are present.")


def get_required_models_list(surrogate: str, conf: Dict) -> list:
    """
    Generate a list of required models based on the configuration settings.

    Args:
        surrogate (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.

    Returns:
        list: A list of required model names.
    """
    required_models = []

    if conf["accuracy"]:
        required_models.append(f"{surrogate.lower()}_main.pth")

    # Dynamic accuracy does not require a separate model
    if conf["dynamic_accuracy"]:
        pass

    if conf["interpolation"]["enabled"]:
        intervals = conf["interpolation"]["intervals"]
        required_models.extend(
            [
                f"{surrogate.lower()}_interpolation_{interval}.pth"
                for interval in intervals
            ]
        )

    if conf["extrapolation"]["enabled"]:
        cutoffs = conf["extrapolation"]["cutoffs"]
        required_models.extend(
            [f"{surrogate.lower()}_extrapolation_{cutoff}.pth" for cutoff in cutoffs]
        )

    if conf["sparse"]["enabled"]:
        factors = conf["sparse"]["factors"]
        required_models.extend(
            [f"{surrogate.lower()}_sparse_{factor}.pth" for factor in factors]
        )

    if conf["UQ"]["enabled"]:
        n_models = conf["UQ"]["n_models"]
        required_models.extend(
            [f"{surrogate.lower()}_UQ_{i}.pth" for i in range(n_models)]
        )

    return required_models


def read_yaml_config(config_path: str) -> Dict:
    """
    Read the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    with open(config_path, "r") as file:
        conf = yaml.safe_load(file)
    return conf


def load_model(
    model, training_id: str, surr_name: str, model_identifier: str
) -> torch.nn.Module:
    """
    Load a trained surrogate model.

    Args:
        model: Instance of the surrogate model class.
        training_id (str): The training identifier.
        surr_name (str): The name of the surrogate model.
        model_identifier (str): The identifier of the model (e.g., 'main').

    Returns:
        The loaded surrogate model.
    """
    statedict_path = os.path.join(
        "trained", surr_name, training_id, f"{model_identifier}.pth"
    )
    model.load_state_dict(torch.load(statedict_path))
    model.eval()
    return model
