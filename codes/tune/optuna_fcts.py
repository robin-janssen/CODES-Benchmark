# optuna_fcts.py
import os
from distutils.util import strtobool

import torch.nn as nn
import yaml

from codes.benchmark.bench_utils import get_surrogate
from codes.utils import check_and_load_data, make_description, set_random_seeds
from codes.utils.data_utils import download_data, get_data_subset


def load_yaml_config(config_path: str) -> dict:
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (str): The path to the YAML config file.

    Returns:
        dict: The configuration dictionary.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_activation_function(name: str) -> nn.Module:
    """
    Convert a string name to a corresponding torch.nn activation function.
    """
    activation_functions = {
        "relu": nn.ReLU(),
        "leakyrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "softplus": nn.Softplus(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity(),
        "elu": nn.ELU(),
    }
    try:
        return activation_functions[name.lower()]
    except KeyError:
        raise ValueError(f"Activation function '{name}' not supported.")


def make_optuna_params(trial, optuna_params):
    """
    Generate suggested parameters for Optuna based on the optuna_params configuration.
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


def create_objective(config, study_name, device_queue):
    """
    Create the Optuna objective function. Each trial:
     - Picks a free device from device_queue
     - Runs training
     - Returns the device to the queue
    """

    def objective(trial):
        # Acquire a free device from the queue
        device = device_queue.get()
        try:
            return training_run(trial, device, config, study_name)
        finally:
            # Return the device to the queue
            device_queue.put(device)

    return objective


def training_run(trial, device, config, study_name):
    """
    Main training procedure for each Optuna trial.
    """
    # Download or load data
    download_data(config["dataset"]["name"])
    train_data, test_data, val_data, timesteps, _, data_params, _ = check_and_load_data(
        config["dataset"]["name"],
        verbose=False,
        log=config["dataset"]["log10_transform"],
        normalisation_mode=config["dataset"]["normalise"],
    )

    # Subset the data if necessary
    subset_factor = config["dataset"].get("subset_factor", 1)
    train_data, test_data, timesteps = get_data_subset(
        train_data, test_data, timesteps, "sparse", subset_factor
    )

    # Set random seed
    set_random_seeds(config["seed"])

    # Prepare model
    surr_name = config["surrogate"]["name"]
    suggested_params = make_optuna_params(trial, config["optuna_params"])

    # Convert or interpret certain parameters
    for key, val in suggested_params.items():
        if "activation" in key:
            suggested_params[key] = get_activation_function(val)
        if "ode_tanh_reg" in key:
            suggested_params[key] = bool(strtobool(val))

    n_timesteps = train_data.shape[1]
    n_chemicals = train_data.shape[2]

    # Instantiate surrogate model
    surrogate_class = get_surrogate(surr_name)
    model = surrogate_class(device, n_chemicals, n_timesteps, suggested_params)

    # Prepare data loaders
    train_loader, test_loader, _ = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=val_data,
        timesteps=timesteps,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    description = make_description("Optuna", device, str(trial.number), surr_name)

    # Train
    model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config["epochs"],
        description=description,
    )

    # Evaluate
    preds, targets = model.predict(test_loader)
    criterion = nn.MSELoss()
    loss = criterion(preds, targets).item()

    # Save model
    # We'll store it in: optuna_runs/studies/<study_name>/models/<ClassName>/
    base_dir = os.path.join("optuna_runs", "studies", study_name, "models")
    model_name = f"{surr_name.lower()}_{trial.number}"
    # training_id="" ensures that your "subfolder" logic in model.save
    # yields the final path: <base_dir>/<ClassName>/<model_name>.pth
    model.save(
        model_name=model_name,
        base_dir=base_dir,
        training_id="",
        data_params=data_params,
    )

    return loss
