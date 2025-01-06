import csv
import importlib.util
import inspect
import os
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
import yaml

from codes.surrogates import SurrogateModel, surrogate_classes


def check_surrogate(surrogate: str, conf: dict) -> None:
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


def check_benchmark(conf: dict) -> None:
    """
    Check whether there are any configuration issues with the benchmark.

    Args:
        conf (dict): The configuration dictionary.

    Raises:
        FileNotFoundError: If the training ID directory is missing or if the .yaml file is missing.
        ValueError: If the configuration is missing required keys or the values do not match the training configuration.
    """
    # Check whether cuda devices have been selected and if they are available
    for device in conf.get("devices", []):
        if "cuda" in device:
            if not torch.cuda.is_available():
                raise ValueError(
                    "You have selected at least one cuda device, but CUDA is not available. Please adjust the device settings in config.yaml."
                )
    # Check for the training directory and load the training configuration
    print("\nChecking benchmark configuration...")
    training_id = conf.get("training_id")
    if not training_id:
        raise ValueError("Configuration must include a 'training_id'.")

    trained_dir = os.path.join(os.getcwd(), "trained", training_id)
    if not os.path.exists(trained_dir):
        raise FileNotFoundError(f"Training ID directory {training_id} not found.")

    yaml_file = os.path.join(trained_dir, "config.yaml")
    if not os.path.isfile(yaml_file):
        raise FileNotFoundError(
            f"Training configuration file not found in directory {trained_dir}."
        )

    with open(yaml_file, "r", encoding="utf-8") as file:
        training_conf = yaml.safe_load(file)

    # 1. Check Surrogates
    training_surrogates = set(training_conf.get("surrogates", []))
    benchmark_surrogates = set(conf.get("surrogates", []))
    if not benchmark_surrogates.issubset(training_surrogates):
        raise ValueError(
            "Benchmark configuration includes surrogates that were not in the training configuration."
        )

    # 2. Check Batch Size
    if "batch_size" in conf:
        training_batch_size = training_conf.get("batch_size", [])
        benchmark_batch_size = conf.get("batch_size", [])

        # Check if batch sizes correspond to the correct surrogates
        for i, surrogate in enumerate(conf.get("surrogates", [])):
            if surrogate in training_conf["surrogates"]:
                index = training_conf["surrogates"].index(surrogate)
                if training_batch_size[index] != benchmark_batch_size[i]:
                    print(
                        f"Warning: Batch size for surrogate '{surrogate}' has changed from {training_batch_size[index]} to {benchmark_batch_size[i]}."
                    )
                    # Get user input to confirm the change
                    user_input = input("Do you want to continue? (y/n): ")
                    if user_input.lower() == "y":
                        print(f"Continuing with batch size {benchmark_batch_size[i]}.")
                    else:
                        print("Exiting...")
                        exit()

    # 3. Check Dataset Settings
    training_dataset = training_conf.get("dataset", {})
    benchmark_dataset = conf.get("dataset", {})

    # Check if any dataset keys or values do not match
    for key, training_value in training_dataset.items():
        benchmark_value = benchmark_dataset.get(key)
        if benchmark_value != training_value:
            raise ValueError(
                f"Dataset setting '{key}' does not match between training and benchmark configurations. "
                f"Training value: {training_value}, Benchmark value: {benchmark_value}."
            )

    # Check if there are any additional keys in the benchmark dataset not present in training
    for key in benchmark_dataset.keys():
        if key not in training_dataset:
            raise ValueError(
                f"Additional dataset setting '{key}' found in benchmark configuration that is not present in training configuration."
            )

    # 4. Check Modalities (Interpolation, Extrapolation, Sparse, Batch Scaling, Uncertainty)
    modalities = [
        "interpolation",
        "extrapolation",
        "sparse",
        "batch_scaling",
        "uncertainty",
    ]
    for modality in modalities:
        training_modality = training_conf.get(modality, {})
        benchmark_modality = conf.get(modality, {})

        # Check if enabled state has changed incorrectly
        if benchmark_modality.get("enabled", False) and not training_modality.get(
            "enabled", False
        ):
            raise ValueError(
                f"Modality '{modality}' is enabled in benchmark but was not enabled in training."
            )

        # Check values within each modality
        if training_modality.get("enabled", False):
            for key, value in benchmark_modality.items():
                if key == "enabled":
                    continue
                if key not in training_modality:
                    raise ValueError(
                        f"Benchmark configuration provides a value for '{key}' in '{modality}' not present in training."
                    )
                if isinstance(value, list):
                    if not set(value).issubset(set(training_modality.get(key, []))):
                        raise ValueError(
                            f"Benchmark configuration provides values for '{key}' in '{modality}' not trained for."
                        )
                else:
                    if modality == "uncertainty" and key == "ensemble_size":
                        if value > training_modality.get(key, value):
                            raise ValueError(
                                f"Benchmark ensemble_size for '{modality}' cannot be larger than in training."
                            )
                    else:
                        if value != training_modality.get(key):
                            raise ValueError(
                                f"Benchmark configuration value for '{key}' in '{modality}' does not match training configuration."
                            )

    print("Configuration check passed successfully.")


def get_required_models_list(surrogate: str, conf: dict) -> list:
    """
    Generate a list of required models based on the configuration settings.

    Args:
        surrogate (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.

    Returns:
        list: A list of required model names.
    """
    required_models = []
    required_models.append(f"{surrogate.lower()}_main.pth")

    # Gradients does not require a separate model
    if conf["gradients"]:
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

    if conf["uncertainty"]["enabled"]:
        n_models = conf["uncertainty"]["ensemble_size"]
        required_models.extend(
            [f"{surrogate.lower()}_UQ_{i+1}.pth" for i in range(n_models - 1)]
        )

    return required_models


def read_yaml_config(config_path: str) -> dict:
    """
    Read the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    with open(config_path, "r", encoding="uft-8") as file:
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
    model: torch.nn.Module,
    inputs: tuple,
) -> dict:
    """
    Measure the memory footprint of the model during the forward and backward pass.

    Args:
        model (torch.nn.Module): The PyTorch model.
        inputs (tuple): The input data for the model.
        conf (dict): The configuration dictionary.
        surr_name (str): The name of the surrogate model.

    Returns:
        dict: A dictionary containing memory footprint measurements.
    """
    # def get_memory_usage():
    #     process = psutil.Process(os.getpid())
    #     return process.memory_info().rss

    def get_memory_usage(model):
        return torch.cuda.memory_allocated(model.device)

    model.to("cpu")

    before_load = get_memory_usage(model)

    model.to(model.device)

    # Measure memory usage before the forward pass
    after_load = get_memory_usage(model)

    inputs = (
        tuple(i.to(model.device) for i in inputs)
        if isinstance(inputs, list) or isinstance(inputs, tuple)
        else inputs.to(model.device)
    )
    before_forward = get_memory_usage(model)
    preds, targets = model(inputs=inputs)
    after_forward = get_memory_usage(model)

    # Measure memory usage before the backward pass
    loss = (preds - targets).sum()  # Example loss function
    before_backward = get_memory_usage(model)
    loss.backward()
    after_backward = get_memory_usage(model)

    del preds, targets, loss

    # Measure pure forward pass memory usage
    model.zero_grad()
    before_forward_nograd = get_memory_usage(model)
    with torch.no_grad():
        preds, targets = model(inputs=inputs)
    after_forward_nograd = get_memory_usage(model)

    memory_usage = {
        "model_memory": after_load - before_load,
        "forward_memory": after_forward - before_forward,
        "backward_memory": after_backward - before_backward,
        "forward_memory_nograd": after_forward_nograd - before_forward_nograd,
    }

    return memory_usage, model


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


def discard_numpy_entries(d: dict) -> dict:
    """
    Recursively remove dictionary entries that contain NumPy arrays.

    Args:
        d (dict): The input dictionary.

    Returns:
        dict: A new dictionary without entries containing NumPy arrays.
    """
    if not isinstance(d, dict):
        return d

    clean_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            cleaned_value = discard_numpy_entries(value)
            if cleaned_value:  # Only add non-empty dictionaries
                clean_dict[key] = cleaned_value
        elif not isinstance(value, np.ndarray):
            clean_dict[key] = value

    return clean_dict


def clean_metrics(metrics: dict, conf: dict) -> dict:
    """
    Clean the metrics dictionary to remove problematic entries.

    Args:
        metrics (dict): The benchmark metrics.
        conf (dict): The configuration dictionary.

    Returns:
        dict: The cleaned metrics dictionary.
    """

    # Make a deep copy of the metrics
    write_metrics = deepcopy(metrics)

    # Remove problematic entries
    write_metrics.pop("timesteps", None)
    write_metrics["accuracy"].pop("absolute_errors", None)
    write_metrics["accuracy"].pop("relative_errors", None)
    if conf["gradients"]:
        write_metrics["gradients"].pop("gradients", None)
        write_metrics["gradients"].pop("max_counts", None)
        write_metrics["gradients"].pop("max_gradient", None)
        write_metrics["gradients"].pop("max_error", None)
    if conf["interpolation"]["enabled"]:
        write_metrics["interpolation"].pop("model_errors", None)
        write_metrics["interpolation"].pop("intervals", None)
    if conf["extrapolation"]["enabled"]:
        write_metrics["extrapolation"].pop("model_errors", None)
        write_metrics["extrapolation"].pop("cutoffs", None)
    if conf["sparse"]["enabled"]:
        write_metrics["sparse"].pop("model_errors", None)
        write_metrics["sparse"].pop("n_train_samples", None)
    if conf["uncertainty"]["enabled"]:
        write_metrics["UQ"].pop("pred_uncertainty", None)
        write_metrics["UQ"].pop("max_counts", None)
        write_metrics["UQ"].pop("axis_max", None)

    return write_metrics


def write_metrics_to_yaml(surr_name: str, conf: dict, metrics: dict) -> None:
    """
    Write the benchmark metrics to a YAML file.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        metrics (dict): The benchmark metrics.
    """
    # Clean the metrics
    write_metrics = clean_metrics(metrics, conf)

    # Convert metrics to standard types
    write_metrics = convert_to_standard_types(write_metrics)

    # Make results directory
    try:
        os.makedirs(f"results/{conf['training_id']}")
    except FileExistsError:
        pass

    with open(
        f"results/{conf['training_id']}/{surr_name.lower()}_metrics.yaml",
        mode="w",
        encoding="utf-8",
    ) as f:
        yaml.dump(write_metrics, f, sort_keys=False)


def get_surrogate(surrogate_name: str) -> SurrogateModel | None:
    """
    Check if the surrogate model exists.

    Args:
        surrogate_name (str): The name of the surrogate model.

    Returns:
        SurrogateModel | None: The surrogate model class if it exists, otherwise None.
    """
    for surrogate in surrogate_classes:
        if surrogate_name == surrogate.__name__:
            return surrogate

    return None


def format_time(mean_time, std_time):
    """
    Format mean and std time consistently in ns, µs, ms, or s.

    Args:
        mean_time: The mean time.
        std_time: The standard deviation of the time.

    Returns:
        str: The formatted time string.
    """
    if mean_time < 1e-6:
        # Both in ns
        return f"{mean_time * 1e9:.2f} ns ± {std_time * 1e9:.2f} ns"
    elif mean_time < 1e-3:
        # Both in µs
        return f"{mean_time * 1e6:.2f} µs ± {std_time * 1e6:.2f} µs"
    elif mean_time < 1:
        # Both in ms
        return f"{mean_time * 1e3:.2f} ms ± {std_time * 1e3:.2f} ms"
    else:
        # Both in s
        return f"{mean_time:.2f} s ± {std_time:.2f} s"


def format_seconds(seconds: int) -> str:
    """
    Format a duration given in seconds as hh:mm:ss.

    Args:
        seconds (int): The duration in seconds.

    Returns:
        str: The formatted duration string.
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def flatten_dict(d: dict, parent_key: str = "", sep: str = " - ") -> dict:
    """
    Flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key string.
        sep (str): The separator between keys.

    Returns:
        dict: Flattened dictionary with composite keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_dict_to_scientific_notation(d: dict, precision: int = 8) -> dict:
    """
    Convert all numerical values in a dictionary to scientific notation.

    Args:
        d (dict): The input dictionary.

    Returns:
        dict: The dictionary with numerical values in scientific notation.
    """
    return {
        k: f"{v:.{precision}e}" if isinstance(v, (int, float)) else v
        for k, v in d.items()
    }


def make_comparison_csv(metrics: dict, config: dict) -> None:
    """
    Generate a CSV file comparing metrics for different surrogate models.

    Args:
        metrics (dict): Dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """

    # Clean the metrics
    for surr_name in metrics.keys():
        metrics[surr_name] = clean_metrics(metrics[surr_name], config)

    # cleaned_metrics = convert_to_standard_types(metrics)

    # Flatten the metrics dictionary for each surrogate model
    flattened_metrics = {
        surr_name: flatten_dict(met) for surr_name, met in metrics.items()
    }

    # Convert all numerical values to scientific notation
    for surr_name in flattened_metrics.keys():
        flattened_metrics[surr_name] = convert_dict_to_scientific_notation(
            flattened_metrics[surr_name]
        )

    # Get all unique keys across all models (i.e., all possible metric categories)
    all_keys = set()
    for surr_name, flat_met in flattened_metrics.items():
        all_keys.update(flat_met.keys())

    # Convert the set of keys to a sorted list
    all_keys = sorted(all_keys)

    # Prepare the CSV file path
    csv_file_path = f"results/{config['training_id']}/metrics.csv"
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Write to the CSV file
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        header = ["Category"] + list(metrics.keys())
        writer.writerow(header)

        # Write each row with the values corresponding to each model
        for key in all_keys:
            row = [key]
            for surr_name in metrics.keys():
                value = flattened_metrics[surr_name].get(
                    key, "N/A"
                )  # Use 'N/A' if key is missing
                row.append(value)
            writer.writerow(row)

    if config["verbose"]:
        print(f"Comparison CSV file saved at {csv_file_path}")


def get_model_config(surr_name: str, config: dict) -> dict:
    """
    Get the model configuration for a specific surrogate model from the dataset folder.
    Returns an empty dictionary if the configuration file is not found.

    Args:
        surr_name (str): The name of the surrogate model.
        config (dict): The configuration dictionary.

    Returns:
        dict: The model configuration dictionary.
    """
    if not config["dataset"].get("use_optimal_params", True):
        return {}

    dataset_name = config["dataset"]["name"].lower()
    dataset_folder = f"datasets/{dataset_name}"
    config_file = f"{dataset_folder}/surrogates_config.py"

    if os.path.exists(config_file):
        spec = importlib.util.spec_from_file_location("config_module", config_file)
        if spec is None:
            raise ImportError(f"Failed to import config module from {config_file}")
        config_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError(f"Failed to load config module from {config_file}")
        spec.loader.exec_module(config_module)

        # Look for the dataclass matching the surr_name + 'Config'
        config_class = None
        target_class_name = f"{surr_name}config".lower()
        for name, obj in inspect.getmembers(config_module, inspect.isclass):
            if (
                hasattr(obj, "__dataclass_fields__")
                and name.lower() == target_class_name
            ):
                config_class = obj
                break

        if config_class:
            # Instantiate the dataclass and convert it to a dictionary
            config_instance = config_class()
            model_config = asdict(config_instance)
        else:
            model_config = {}
    else:
        model_config = {}

    return model_config
