import os
import yaml
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
    base_dir = os.path.join("models", surrogate, "trained", training_id)

    required_models = get_required_models_list(surrogate, conf)

    for model_name in required_models:
        model_path = os.path.join(base_dir, f"{model_name}.pth")
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

    if conf.get("accuracy", False):
        required_models.append(f"main_{surrogate.lower()}")

    if conf.get("interpolation", {}).get("enabled", False):
        intervals = conf["interpolation"]["intervals"]
        required_models.extend(
            [f"{surrogate.lower()}_interpolation_{interval}" for interval in intervals]
        )

    if conf.get("extrapolation", {}).get("enabled", False):
        cutoffs = conf["extrapolation"]["cutoffs"]
        required_models.extend(
            [f"{surrogate.lower()}_extrapolation_{cutoff}" for cutoff in cutoffs]
        )

    if conf.get("sparse", {}).get("enabled", False):
        factors = conf["sparse"]["factors"]
        required_models.extend(
            [f"{surrogate.lower()}_sparse_{factor}" for factor in factors]
        )

    if conf.get("UQ", {}).get("enabled", False):
        n_models = conf["UQ"]["n_models"]
        required_models.extend(
            [f"{surrogate.lower()}_ensemble_{i}" for i in range(n_models)]
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
