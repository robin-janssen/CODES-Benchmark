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
    Suggest hyperparameters from optuna_params, sampling conditional parameters:
      - coeff_width / coeff_layers only if coeff_network is True
      - momentum only if optimizer == "sgd"
      - eta_min only if scheduler == "cosine"
      - poly_power only if scheduler == "poly"
    Also converts "True"/"False" strings to bool and maps activation names.
    """
    suggested = {}

    # Define switch keys and their children
    coeff_switch = "coeff_network"
    coeff_children = {"coeff_width", "coeff_layers"}

    optimizer_switch = "optimizer"
    optimizer_children = {"momentum"}

    scheduler_switch = "scheduler"
    scheduler_children = {"eta_min", "poly_power"}

    # Sample all independent params (skip all switch keys and their children)
    skip_keys = (
        {coeff_switch, optimizer_switch, scheduler_switch}
        | coeff_children
        | optimizer_children
        | scheduler_children
    )
    for name, opts in optuna_params.items():
        if name in skip_keys:
            continue
        if opts["type"] == "int":
            suggested[name] = trial.suggest_int(
                name, opts["low"], opts["high"], step=opts.get("step", 1)
            )
        elif opts["type"] == "float":
            suggested[name] = trial.suggest_float(
                name, opts["low"], opts["high"], log=opts.get("log", False)
            )
        else:  # categorical
            raw = trial.suggest_categorical(name, opts["choices"])
            # convert string booleans if needed
            if isinstance(raw, str) and raw.lower() in ("true", "false"):
                suggested[name] = bool(strtobool(raw))
            else:
                suggested[name] = raw

    # Sample coeff_network and its children
    if coeff_switch in optuna_params:
        opts = optuna_params[coeff_switch]
        raw = trial.suggest_categorical(coeff_switch, opts["choices"])
        if isinstance(raw, str) and raw.lower() in ("true", "false"):
            suggested[coeff_switch] = bool(strtobool(raw))
        else:
            suggested[coeff_switch] = raw
    if suggested.get(coeff_switch, False):
        for child in coeff_children:
            if child in optuna_params:
                opts = optuna_params[child]
                if opts["type"] == "int":
                    suggested[child] = trial.suggest_int(
                        child, opts["low"], opts["high"], step=opts.get("step", 1)
                    )
                elif opts["type"] == "float":
                    suggested[child] = trial.suggest_float(
                        child, opts["low"], opts["high"], log=opts.get("log", False)
                    )
                else:
                    raw = trial.suggest_categorical(child, opts["choices"])
                    if isinstance(raw, str) and raw.lower() in ("true", "false"):
                        suggested[child] = bool(strtobool(raw))
                    else:
                        suggested[child] = raw

    # Sample optimizer and its dependent momentum
    if optimizer_switch in optuna_params:
        opts = optuna_params[optimizer_switch]
        raw = trial.suggest_categorical(optimizer_switch, opts["choices"])
        # normalize to lowercase string
        opt_choice = raw.lower() if isinstance(raw, str) else raw
        suggested[optimizer_switch] = opt_choice
        if opt_choice == "sgd" and "momentum" in optuna_params:
            mopts = optuna_params["momentum"]
            if mopts["type"] == "int":
                suggested["momentum"] = trial.suggest_int(
                    "momentum", mopts["low"], mopts["high"], step=mopts.get("step", 1)
                )
            elif mopts["type"] == "float":
                suggested["momentum"] = trial.suggest_float(
                    "momentum", mopts["low"], mopts["high"], log=mopts.get("log", False)
                )
            else:
                rawm = trial.suggest_categorical("momentum", mopts["choices"])
                if isinstance(rawm, str) and rawm.lower() in ("true", "false"):
                    suggested["momentum"] = bool(strtobool(rawm))
                else:
                    suggested["momentum"] = rawm

    # Sample scheduler and its dependent eta_min / poly_power
    if scheduler_switch in optuna_params:
        opts = optuna_params[scheduler_switch]
        raw = trial.suggest_categorical(scheduler_switch, opts["choices"])
        sched_choice = raw.lower() if isinstance(raw, str) else raw
        suggested[scheduler_switch] = sched_choice
        if sched_choice == "cosine" and "eta_min" in optuna_params:
            eopts = optuna_params["eta_min"]
            if eopts["type"] == "int":
                suggested["eta_min"] = trial.suggest_int(
                    "eta_min", eopts["low"], eopts["high"], step=eopts.get("step", 1)
                )
            elif eopts["type"] == "float":
                suggested["eta_min"] = trial.suggest_float(
                    "eta_min", eopts["low"], eopts["high"], log=eopts.get("log", False)
                )
            else:
                raw_e = trial.suggest_categorical("eta_min", eopts["choices"])
                if isinstance(raw_e, str) and raw_e.lower() in ("true", "false"):
                    suggested["eta_min"] = bool(strtobool(raw_e))
                else:
                    suggested["eta_min"] = raw_e
        elif sched_choice == "poly" and "poly_power" in optuna_params:
            popts = optuna_params["poly_power"]
            if popts["type"] == "int":
                suggested["poly_power"] = trial.suggest_int(
                    "poly_power", popts["low"], popts["high"], step=popts.get("step", 1)
                )
            elif popts["type"] == "float":
                suggested["poly_power"] = trial.suggest_float(
                    "poly_power",
                    popts["low"],
                    popts["high"],
                    log=popts.get("log", False),
                )
            else:
                raw_p = trial.suggest_categorical("poly_power", popts["choices"])
                if isinstance(raw_p, str) and raw_p.lower() in ("true", "false"):
                    suggested["poly_power"] = bool(strtobool(raw_p))
                else:
                    suggested["poly_power"] = raw_p

    # Post-process: convert any remaining "True"/"False" strings to bool, and map activations
    for name, val in list(suggested.items()):
        if isinstance(val, str) and val.lower() in ("true", "false"):
            suggested[name] = bool(strtobool(val))
        if "activation" in name.lower():
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
