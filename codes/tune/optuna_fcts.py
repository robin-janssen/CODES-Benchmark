import os
import queue
from distutils.util import strtobool

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml

from codes.benchmark.bench_utils import (
    get_model_config,
    get_surrogate,
    measure_inference_time,
)
from codes.utils import check_and_load_data, make_description, set_random_seeds
from codes.utils.data_utils import download_data, get_data_subset


def load_yaml_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_activation_function(name: str) -> nn.Module:
    """
    Get the activation function module from its name.
    Required for Optuna to suggest activation functions.

    Args:
        name (str): Name of the activation function.

    Returns:
        nn.Module: Activation function module.
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
    return activation_functions[name.lower()]


def make_optuna_params(trial: optuna.Trial, optuna_params: dict) -> dict:
    """
    Make Optuna suggested parameters from the optuna_config.yaml file.

    Args:
        trial (optuna.Trial): Optuna trial object.
        optuna_params (dict): Optuna parameters dictionary.

    Returns:
        dict: Suggested parameters.
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


def create_objective(
    config: dict, study_name: str, device_queue: queue.Queue
) -> callable:
    """
    Create the objective function for Optuna.

    Args:
        config (dict): Configuration dictionary.
        study_name (str): Name of the study.
        device_queue (queue.Queue): Queue of available devices.

    Returns:
        function: Objective function for Optuna.
    """

    def objective(trial):
        device = device_queue.get()
        try:
            try:
                return training_run(trial, device, config, study_name)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                msg = repr(e).strip()
                if not msg:
                    msg = "CUDA Out of Memory (no details provided)."
                print(f"Trial {trial.number} failed due to: {msg}")
                trial.set_user_attr("exception", msg)
                raise optuna.TrialPruned(f"OOM error in trial {trial.number}")
            except optuna.TrialPruned as e:
                msg = repr(e).strip()
                trial.set_user_attr("exception", msg)
                raise
            except Exception as e:
                torch.cuda.empty_cache()
                msg = repr(e).strip()
                if not msg:
                    msg = "Unknown error occurred."
                print(f"Trial {trial.number} failed due to an unexpected error: {msg}")
                trial.set_user_attr("exception", msg)
                raise optuna.TrialPruned(f"Error in trial {trial.number}: {msg}")
        finally:
            device_queue.put(device)

    return objective


def training_run(
    trial: optuna.Trial, device: str, config: dict, study_name: str
) -> float | tuple[float, float]:
    """
    Run the training for a single Optuna trial and return the loss.
    In multi-objective mode, also returns the mean inference time.

    Args:
        trial (optuna.Trial): Optuna trial object.
        device (str): Device to run the training on.
        config (dict): Configuration dictionary.
        study_name (str): Name of the study.

    Returns:
        float: Loss value in single objective mode.
        tuple[float, float]: (loss, mean_inference_time) in multi objective mode.
    """

    download_data(config["dataset"]["name"], verbose=False)
    train_data, test_data, val_data, timesteps, _, data_info, _ = check_and_load_data(
        config["dataset"]["name"],
        verbose=False,
        log=config["dataset"]["log10_transform"],
        normalisation_mode=config["dataset"]["normalise"],
    )

    # Load full data and parameters
    (
        (train_data, test_data, val_data),
        _,
        timesteps,
        _,
        _,
        _,
    ) = check_and_load_data(
        config["dataset"]["name"],
        verbose=False,
        log=config["dataset"]["log10_transform"],
        normalisation_mode=config["dataset"]["normalise"],
        tolerance=config["dataset"]["tolerance"],
    )

    subset_factor = config["dataset"].get("subset_factor", 1)
    train_data, test_data, timesteps = get_data_subset(
        train_data, test_data, timesteps, "sparse", subset_factor
    )

    set_random_seeds(config["seed"], device=device)
    surr_name = config["surrogate"]["name"]
    suggested_params = make_optuna_params(trial, config["optuna_params"])

    for key, val in suggested_params.items():
        if "activation" in key:
            suggested_params[key] = get_activation_function(val)
        if "ode_tanh_reg" in key:
            suggested_params[key] = bool(strtobool(val))

    n_timesteps = train_data.shape[1]
    n_quantities = train_data.shape[2]
    surrogate_class = get_surrogate(surr_name)
    model_config = get_model_config(surr_name, config)
    model_config.update(suggested_params)
    model = surrogate_class(device, n_quantities, n_timesteps, model_config)
    model.normalisation = data_info
    model.optuna_trial = trial
    model.trial_update_epochs = 10

    train_loader, test_loader, _ = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=val_data,
        timesteps=timesteps,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    description = make_description("Optuna", device, str(trial.number), surr_name)
    pos = config["devices"].index(device) + 2
    model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config["epochs"],
        position=pos,
        description=description,
        multi_objective=config["multi_objective"],
    )

    criterion = torch.nn.MSELoss()
    preds, targets = model.predict(test_loader)
    loss = criterion(preds, targets).item()
    sname, _ = study_name.split("_")

    savepath = os.path.join("tuned", sname, "models")
    os.makedirs(savepath, exist_ok=True)
    model_name = f"{surr_name.lower()}_{trial.number}"
    model.save(
        model_name=model_name,
        base_dir="",
        training_id=savepath,
    )

    # Check if we're running multi-objective optimisation
    if config["multi_objective"]:
        # Measure inference time
        inference_times = measure_inference_time(model, test_loader)
        return loss, np.mean(inference_times)
    else:
        return loss
