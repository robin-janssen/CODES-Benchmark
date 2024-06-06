import os
import numpy as np
import yaml
import torch
import psutil
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
    training_id = conf["training_id"]
    base_dir = os.getcwd()
    base_dir = os.path.join(base_dir, "trained", training_id, surrogate)

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
            [f"{surrogate.lower()}_UQ_{i+1}.pth" for i in range(n_models - 1)]
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
        "trained", training_id, surr_name, f"{model_identifier}.pth"
    )
    model.load_state_dict(torch.load(statedict_path))
    model.eval()
    return model


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_memory_footprint(
    model: torch.nn.Module, initial_conditions: torch.Tensor, times: torch.Tensor
) -> dict:
    """
    Measure the memory footprint of the model during the forward and backward pass.

    Args:
        model (torch.nn.Module): The PyTorch model.
        initial_conditions (torch.Tensor): The initial conditions tensor.
        times (torch.Tensor): The times tensor.

    Returns:
        dict: A dictionary containing memory footprint measurements.
    """
    # model.to(model.device)
    initial_conditions = initial_conditions.to(model.device)
    times = times.to(model.device)

    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    # Measure memory usage before the forward pass
    before_forward = get_memory_usage()

    # Forward pass
    output = model(initial_conditions, times)
    after_forward = get_memory_usage()

    # Measure memory usage before the backward pass
    loss = output.sum()  # Example loss function
    loss.backward()
    after_backward = get_memory_usage()

    return {
        "before_forward": before_forward,
        "after_forward": after_forward,
        "forward_pass_memory": after_forward - before_forward,
        "after_backward": after_backward,
        "backward_pass_memory": after_backward - after_forward,
    }


def convert_to_standard_types(data):
    """
    Recursively convert data to standard types that can be serialized to YAML.

    Args:
        data: The data to convert.

    Returns:
        The converted data.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {k: convert_to_standard_types(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_to_standard_types(i) for i in data]
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        return str(data)


def write_metrics_to_yaml(surr_name: str, conf: dict, metrics: dict) -> None:
    """
    Write the benchmark metrics to a YAML file.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        metrics (dict): The benchmark metrics.
    """
    # Convert metrics to standard types
    metrics = convert_to_standard_types(metrics)

    # Make results directory
    try:
        os.makedirs(f"results/{conf['training_id']}")
    except FileExistsError:
        pass

    with open(
        f"results/{conf['training_id']}/{surr_name.lower()}_metrics.yaml", "w"
    ) as f:
        yaml.dump(metrics, f, sort_keys=False)
