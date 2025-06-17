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
        "prelu": nn.PReLU(),
        "mish": nn.Mish(),
        "silu": nn.SiLU(),
    }
    return activation_functions[name.lower()]


def make_optuna_params(trial: optuna.Trial, optuna_params: dict) -> dict:
    """
    Suggest hyperparameters from optuna_params, sampling coeff_width/layers
    only if coeff_network == True, converting any "True"/"False" strings into bool,
    and mapping any activation names into nn.Modules.
    """
    suggested = {}

    switch_key = "coeff_network"
    children = {"coeff_width", "coeff_layers"}

    # 1) Sample all independent params (skip switch & its children)
    skip = children | {switch_key}
    for name, opts in optuna_params.items():
        if name in skip:
            continue
        if opts["type"] == "int":
            suggested[name] = trial.suggest_int(name, opts["low"], opts["high"])
        elif opts["type"] == "float":
            suggested[name] = trial.suggest_float(
                name, opts["low"], opts["high"], log=opts.get("log", False)
            )
        else:  # categorical
            suggested[name] = trial.suggest_categorical(name, opts["choices"])

    # 2) Sample the coeff_network switch
    if switch_key in optuna_params:
        opts = optuna_params[switch_key]
        raw = trial.suggest_categorical(switch_key, opts["choices"])
        # convert string → bool if needed
        if isinstance(raw, str) and raw.lower() in ("true", "false"):
            suggested[switch_key] = bool(strtobool(raw))
        else:
            suggested[switch_key] = raw

    # 3) Conditionally sample coeff_width & coeff_layers
    if suggested.get(switch_key, False):
        for child in children:
            if child in optuna_params:
                opts = optuna_params[child]
                if opts["type"] == "int":
                    suggested[child] = trial.suggest_int(
                        child, opts["low"], opts["high"]
                    )
                elif opts["type"] == "float":
                    suggested[child] = trial.suggest_float(
                        child, opts["low"], opts["high"], log=opts.get("log", False)
                    )
                else:
                    suggested[child] = trial.suggest_categorical(child, opts["choices"])

    # 4) Post‐process all values: booleans and activation modules
    for name, val in list(suggested.items()):
        # a) convert any remaining "True"/"False" strings into bool
        if isinstance(val, str) and val.lower() in ("true", "false"):
            suggested[name] = bool(strtobool(val))
        # b) map activation names to nn.Modules
        if "activation" in name:
            suggested[name] = get_activation_function(suggested[name])

    return suggested


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

    # Load full data and parameters
    (
        (train_data, test_data, _),
        (train_params, test_params, _),
        timesteps,
        _,
        data_info,
        _,
    ) = check_and_load_data(
        config["dataset"]["name"],
        verbose=False,
        log=config["dataset"]["log10_transform"],
        log_params=config.get("log10_transform_params", False),
        normalisation_mode=config["dataset"]["normalise"],
        tolerance=config["dataset"]["tolerance"],
    )

    subset_factor = config["dataset"].get("subset_factor", 1)
    # Get the appropriate data subset
    (train_data, test_data), (train_params, test_params), timesteps = get_data_subset(
        (train_data, test_data),
        timesteps,
        "sparse",
        subset_factor,
        (train_params, test_params),
    )

    set_random_seeds(config["seed"], device=device)
    surr_name = config["surrogate"]["name"]
    suggested_params = make_optuna_params(trial, config["optuna_params"])
    n_params = train_params.shape[1] if train_params is not None else 0

    n_timesteps = train_data.shape[1]
    n_quantities = train_data.shape[2]
    surrogate_class = get_surrogate(surr_name)
    model_config = get_model_config(surr_name, config)
    model_config.update(suggested_params)
    model = surrogate_class(
        device=device,
        n_quantities=n_quantities,
        n_timesteps=n_timesteps,
        n_parameters=n_params,
        config=model_config,
    )
    model.normalisation = data_info
    model.optuna_trial = trial
    model.trial_update_epochs = 10

    train_loader, test_loader, _ = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=None,
        timesteps=timesteps,
        batch_size=config["batch_size"],
        shuffle=True,
        dataset_train_params=train_params,
        dataset_test_params=test_params,
        dummy_timesteps=True,
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
