import yaml
import functools
import time
import os


def read_yaml_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def time_execution(func):
    """
    Decorator to time the execution of a function and store the duration
    as an attribute of the function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.duration = end_time - start_time
        print(f"{func.__name__} executed in {wrapper.duration:.2f} seconds.")
        return result

    wrapper.duration = None
    return wrapper


def create_model_dir(
    base_dir: str = ".", subfolder: str = "models", unique_id: str = "run_1"
) -> str:
    """
    Create a directory based on a unique identifier inside a specified subfolder of the base directory.

    :param base_dir: The base directory where the subfolder and unique directory will be created.
    :param subfolder: The subfolder inside the base directory to include before the unique directory.
    :param unique_id: A unique identifier to be included in the directory name.
    :return: The path of the created unique directory within the specified subfolder.
    """
    unique_dir_name = unique_id
    full_path = os.path.join(base_dir, subfolder, unique_dir_name)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path
