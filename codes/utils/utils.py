import functools
import json
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm


def read_yaml_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def time_execution(func):
    """
    Decorator to time the execution of a function and store the duration
    as an attribute of the function.

    Args:
        func (callable): The function to be timed.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.duration = end_time - start_time
        # tqdm.write(f"{func.__name__} executed in {wrapper.duration:.2f} seconds.")
        return result

    wrapper.duration = None
    return wrapper


def create_model_dir(
    base_dir: str = ".", subfolder: str = "trained", unique_id: str = ""
) -> str:
    """
    Create a directory based on a unique identifier inside a specified subfolder of the base directory.

    Args:
        base_dir (str): The base directory where the subfolder and unique directory will be created.
        subfolder (str): The subfolder inside the base directory to include before the unique directory.
        unique_id (str): A unique identifier to be included in the directory name.

    Returns:
        str: The path of the created unique directory within the specified subfolder.
    """
    full_path = os.path.join(base_dir, subfolder, unique_id)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def load_and_save_config(config_path: str = "config.yaml", save: bool = True) -> dict:
    """
    Load configuration from a YAML file and save a copy to the specified directory.

    Args:
        config_path (str): The path to the configuration YAML file.
        save (bool): Whether to save a copy of the configuration file. Default is True.

    Returns:
        dict: The loaded configuration dictionary.
    """
    # Load configuration from YAML
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if save:
        # Get training ID from the config
        training_id = config["training_id"]

        # Create the directory if it does not exist
        save_dir = os.path.join("trained", training_id)
        os.makedirs(save_dir, exist_ok=True)

        # Copy the config file to the directory
        config_save_path = os.path.join(save_dir, "config.yaml")
        shutil.copyfile(config_path, config_save_path)

    return config


def set_random_seeds(seed: int, device: str) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The random seed to set.
        device (str): The device to use for training, e.g., 'cuda:0'
    """
    # Set the device explicitly in case of GPU:
    if "cuda" in device:
        torch.cuda.device(torch.device(device))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nice_print(message: str, width: int = 80) -> None:
    """
    Print a message in a nicely formatted way with a fixed width.

    Args:
        message (str): The message to print.
        width (int): The width of the printed box. Default is 80.
    """
    # Calculate padding
    padding = (width - len(message) - 2) // 2
    padding_left = padding
    padding_right = padding

    # If message length is odd, add one more space to the right
    if (width - len(message)) % 2 != 0:
        padding_right += 1

    border = "-" * width
    print(
        f"\n{border}\n|{' ' * padding_left}{message}{' ' * padding_right}|\n{border}\n"
    )


def make_description(mode: str, device: str, metric: str, surrogate_name: str) -> str:
    """
    Create a formatted description for the progress bar that ensures consistent alignment.

    Args:
        mode (str): The benchmark mode (e.g., "accuracy", "interpolation", "extrapolation", "sparse", "UQ").
        device (str): The device to use for training (e.g., 'cuda:0').
        metric (str): The specific metric for the mode (e.g., interval, cutoff, factor, batch size).
        surrogate_name (str): The name of the surrogate model.

    Returns:
        str: A formatted description string for the progress bar.
    """
    surrogate_name = surrogate_name.ljust(14)
    mode = mode.ljust(13)
    metric = metric.ljust(2)
    if device == "":
        device = device.ljust(6)
    else:
        device = f"({device})".ljust(8)

    description = f"{surrogate_name} {mode} {metric} {device}"
    return description


def get_progress_bar(tasks: list) -> tqdm:
    """
    Create a progress bar with a specific description.

    Args:
        tasks (list): The list of tasks to be executed.

    Returns:
        tqdm: The created progress bar.
    """
    overall_progress_bar = tqdm(
        total=len(tasks),
        desc=make_description("", "", "", "Overall Progress"),
        position=0,
        leave=True,
        bar_format="{l_bar}{bar} | {n_fmt:>3}/{total_fmt} models trained [elapsed time: {elapsed}]",
    )
    return overall_progress_bar


def worker_init_fn(worker_id):
    """
    Initialize the random seed for each worker in PyTorch DataLoader.

    Args:
        worker_id (int): The worker ID.
    """
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32 - 1
    np.random.seed(np_seed)


def save_task_list(tasks: list, filepath: str) -> None:
    """
    Save a list of tasks to a JSON file.

    Args:
        tasks (list): The list of tasks to save.
        filepath (str): The path to the JSON file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(tasks, f)


def load_task_list(filepath: str | None) -> list:
    """
    Load a list of tasks from a JSON file.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        list: The loaded list of tasks
    """
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        return tasks
    else:
        return []


def check_training_status(config: dict) -> tuple[str, bool]:
    """
    Check if the training is already completed by looking for a completion marker file.
    If the training is not complete, compare the configurations and ask for a confirmation if there are differences.

    Args:
        config (dict): The configuration dictionary.
    Returns:
        str: The path to the task list file.
        bool: Whether to copy the configuration file.
    """
    training_id = config["training_id"]
    task_list_filepath = os.path.join(f"trained/{training_id}/train_tasks.json")
    completion_marker_filepath = os.path.join(f"trained/{training_id}/completed.txt")

    # Check if training is already complete
    if os.path.exists(completion_marker_filepath):
        print()
        print("Training is already completed. Exiting. \n")
        sys.exit()
    else:
        # Check if the configuration is different from the saved configuration
        saved_config_path = os.path.join(f"trained/{training_id}/config.yaml")
        if not os.path.exists(saved_config_path):
            return task_list_filepath, True

        saved_config = read_yaml_config(saved_config_path)

        # Check if cuda is available if the device is set to cuda
        for device in config["devices"]:
            if "cuda" in device:
                if not torch.cuda.is_available():
                    raise ValueError(
                        "You have selected at least one cuda device, but CUDA is not available. Please adjust the device settings in config.yaml."
                    )

        # Check if the configurations are the same
        errors = []
        for key, value in config.items():
            if key not in saved_config:
                errors.append(f"Key '{key}' not found in saved configuration.")
            elif value != saved_config[key]:
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key not in saved_config[key]:
                            errors.append(
                                f"Key '{sub_key}' not found in '{key}' of saved configuration."
                            )
                        elif sub_value != saved_config[key][sub_key]:
                            errors.append(
                                f"Value of '{sub_key}' in '{key}' is different from saved configuration:\n"
                                f"    Previous: {saved_config[key][sub_key]}, current: {sub_value}"
                            )
                else:
                    errors.append(
                        f"Value of '{key}' is different from saved configuration:\n"
                        f"    Previous: {saved_config[key]}, current: {value}"
                    )

        print()
        if len(errors) > 0:
            print("The current configuration differs from the saved configuration:")
            for error in errors:
                print(f"  - {error}")
            print(
                "You can overwrite the saved configuration or resume the training with the previous configuration."
            )
            confirmation = input("Overwrite? [y/n]: ")
            if confirmation.lower() == "y":
                print("Overwriting the saved configuration.")
                if os.path.exists(task_list_filepath):
                    os.remove(task_list_filepath)
                copy_config = True
            else:
                print("Continuing training with the previous configuration.")
                nice_print("Resuming training.")
                copy_config = False
        else:
            nice_print("Resuming training.")
            copy_config = True

    return task_list_filepath, copy_config


def determine_batch_size(config, surr_idx, mode, metric):
    """
    Determine the appropriate batch size based on the config, surrogate index, mode, and metric.

    Args:
        config (dict): The configuration dictionary.
        surr_idx (int): Index of the surrogate model in the config.
        mode (str): The benchmark mode (e.g., "main", "batch_size").
        metric (int): Metric used for determining the batch size in "batch_size" mode.

    Returns:
        int: The determined batch size.

    Raises:
        ValueError: If the number of batch sizes does not match the number of surrogates.
    """
    batch_size_config = config["batch_size"]
    if isinstance(batch_size_config, list):
        if len(batch_size_config) != len(config["surrogates"]):
            raise ValueError(
                "The number of provided batch sizes must match the number of surrogate models."
            )
        batch_size = batch_size_config[surr_idx]
    else:
        batch_size = batch_size_config

    return metric if mode == "batch_size" else batch_size


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
