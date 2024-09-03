import importlib.util
import os

import yaml
from torch import nn

from benchmark.bench_utils import get_surrogate
from data import check_and_load_data
from data.data_utils import download_data, get_data_subset
from utils import make_description, set_random_seeds


def load_config_from_pyfile(pyfile_name):
    """
    Dynamically loads a configuration dictionary from a given .py file.

    Args:
        pyfile_name (str): The name of the .py file containing the configuration dictionary.

    Returns:
        dict: The configuration dictionary from the .py file.
    """
    conf_path = f"optuna_runs/configs/{pyfile_name}"
    spec = importlib.util.spec_from_file_location("config_module", conf_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def save_optuna_config(config, study_name, folder_path="optuna_runs/studies/"):
    """
    Function to save the Optuna configuration to a YAML file.

    Args:
        config (dict): The configuration dictionary to be saved.
        study_name (str): The name of the study.
        folder_path (str): The folder path where the YAML file will be saved.
    """
    # Create directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Define the file path
    file_path = os.path.join(folder_path, f"{study_name}_config.yaml")

    # Increase number of the name if file already exists until it doesn't
    if os.path.exists(file_path):
        i = 1
        new_file_path = os.path.join(folder_path, f"{study_name}_{i}_config.yaml")
        while os.path.exists(new_file_path):
            i += 1
            new_file_path = os.path.join(folder_path, f"{study_name}_{i}_config.yaml")
        file_path = new_file_path

    # Save the configuration as a YAML file
    with open(file_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def get_activation_function(model_dict):
    """
    Get and replace activation functions in the model dictionary.

    Args:
        model_dict (dict): The model dictionary containing the activation functions.

    Returns:
        model_dict (dict): The model dictionary with the activation functions replaced by the corresponding PyTorch functions.
    """

    def get_activation(activation):
        activation = activation.strip().lower()
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leakyrelu":
            return nn.LeakyReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "softplus":
            return nn.Softplus()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "identity":
            return nn.Identity()
        elif activation == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Activation function {activation} not supported.")

    if "activation" in model_dict:
        for key in model_dict:
            if "activation" in key:
                model_dict[key] = get_activation(model_dict[key])

    return model_dict


def make_optuna_params(trial, optuna_params):
    """
    Function to generate suggested parameters for Optuna based on the optuna_params configuration.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object.
        optuna_params (dict): A dictionary specifying the hyperparameters to tune and their ranges.

    Returns:
        dict: A dictionary of suggested hyperparameters.
    """
    suggested_params = {}
    for param_name, param_options in optuna_params.items():
        if param_options["type"] == "int":
            suggested_params[param_name] = trial.suggest_int(
                param_name, param_options["low"], param_options["high"]
            )
        elif param_options["type"] == "float":
            suggested_params[param_name] = trial.suggest_float(
                param_name,
                param_options["low"],
                param_options["high"],
                log=param_options.get("log", False),
            )
        elif param_options["type"] == "categorical":
            suggested_params[param_name] = trial.suggest_categorical(
                param_name, param_options["choices"]
            )
    return suggested_params


def create_objective(config):
    """
    Function to create the Optuna objective function with a specific config.

    Args:
        config (dict): The configuration dictionary to be used within the objective function.

    Returns:
        function: The objective function for Optuna with config captured.
    """

    def objective(trial):
        return training_run(trial, config)

    return objective


def training_run(trial, config):
    # Load data
    download_data(config["dataset"]["name"])
    train_data, test_data, val_data, timesteps, _, data_params, _ = check_and_load_data(
        config["dataset"]["name"],
        verbose=False,
        log=config["dataset"]["log10_transform"],
        normalisation_mode=config["dataset"]["normalise"],
    )

    subset_factor = config["dataset"].get("subset_factor", 1)  # Default is 1

    train_data, test_data, timesteps = get_data_subset(
        train_data, test_data, timesteps, "sparse", subset_factor
    )

    set_random_seeds(config["seed"])

    surr_name = config["surrogate"]["name"]

    # Generate suggested parameters using the make_optuna_params function
    suggested_params = make_optuna_params(trial, config["optuna_params"])
    # See whether "activation" is part of any of the strings in the suggested_params
    if "activation" in suggested_params:
        for key in suggested_params:
            if "activation" in key:
                suggested_params[key] = get_activation_function(suggested_params[key])
    n_timesteps = train_data.shape[1]
    n_chemicals = train_data.shape[2]

    # Get the surrogate class
    surrogate_class = get_surrogate(surr_name)

    model = surrogate_class(
        config["device"], n_chemicals, n_timesteps, suggested_params
    )

    train_loader, test_loader, _ = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=val_data,
        timesteps=timesteps,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    description = make_description(
        "Optuna", config["device"], str(trial.number), surr_name
    )

    # Train the model
    model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config["epochs"],
        description=description,
    )

    preds, targets = model.predict(test_loader)
    criterion = nn.MSELoss()
    loss = criterion(preds, targets).item()

    surrogate_name = surr_name.lower()
    model_name = f"{surrogate_name}_{str(trial.number)}".strip("_")
    model_name = model_name.replace("__", "_")
    model.save(
        model_name=model_name,
        training_id=config["dataset"]["name"],
        subfolder="optuna_runs/models/",
        data_params=data_params,
    )

    return loss
