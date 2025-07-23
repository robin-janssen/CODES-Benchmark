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

MODULE_REGISTRY: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
    "mish": nn.Mish,
    "silu": nn.SiLU,
    "mse": nn.MSELoss,
    "smoothl1": nn.SmoothL1Loss,
}


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


def _suggest_param(trial: optuna.Trial, name: str, opts: dict):
    """Suggest a single parameter, handling int, float, categorical and boolean conversion."""
    param_type = opts.get("type", "categorical")
    if param_type == "int":
        return trial.suggest_int(
            name, opts["low"], opts["high"], step=opts.get("step", 1)
        )
    if param_type == "float":
        return trial.suggest_float(
            name, opts["low"], opts["high"], log=opts.get("log", False)
        )
    raw = trial.suggest_categorical(name, opts.get("choices", []))
    if isinstance(raw, str) and raw.lower() in ("true", "false"):
        return bool(strtobool(raw))
    return raw


def make_optuna_params(trial: optuna.Trial, optuna_params: dict) -> dict:
    suggested = {}

    # helper to sample switch and its children
    def _handle_switch(switch, children=()):
        if switch not in optuna_params:
            return
        suggested[switch] = _suggest_param(trial, switch, optuna_params[switch])
        if suggested.get(switch) and children:
            for child in children:
                if child in optuna_params:
                    suggested[child] = _suggest_param(
                        trial, child, optuna_params[child]
                    )

    # sample independent params
    skip = set(optuna_params) & {
        "coeff_network",
        "optimizer",
        "scheduler",
        "loss_function",
    }
    skip |= {"coeff_width", "coeff_layers", "momentum", "eta_min", "poly_power", "beta"}
    for name, opts in optuna_params.items():
        if name in skip:
            continue
        suggested[name] = _suggest_param(trial, name, opts)

    _handle_switch("coeff_network", ("coeff_width", "coeff_layers"))
    _handle_switch("optimizer", ("momentum",))
    _handle_switch("scheduler", ("eta_min", "poly_power"))
    _handle_switch("loss_function", ("beta",))

    # map activations and loss functions
    for name, val in list(suggested.items()):
        if "activation" in name.lower():
            if val.lower() in MODULE_REGISTRY:
                suggested[name] = MODULE_REGISTRY[val.lower()]()
            else:
                raise ValueError(f"Unknown activation function: {val}")
        elif name == "loss_function":
            if val.lower() in MODULE_REGISTRY:
                suggested[name] = MODULE_REGISTRY[val.lower()]()
            else:
                raise ValueError(f"Unknown loss function: {val}")

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
        device, slot_id = device_queue.get()
        try:
            try:
                return training_run(trial, device, slot_id, config, study_name)
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
            device_queue.put((device, slot_id))

    return objective


def training_run(
    trial: optuna.Trial, device: str, slot_id: int, config: dict, study_name: str
) -> float | tuple[float, float]:
    """
    Run the training for a single Optuna trial and return the loss.
    In multi-objective mode, also returns the mean inference time.

    Args:
        trial (optuna.Trial): Optuna trial object.
        device (str): Device to run the training on.
        slot_id (int): Slot ID for the position of the progress bar.
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
        per_species=config["dataset"].get("normalise_per_species", False),
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
    model.to(device)
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
    model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config["epochs"],
        position=slot_id + 2,
        description=description,
        multi_objective=config["multi_objective"],
    )

    # criterion = torch.nn.MSELoss()
    preds, targets = model.predict(test_loader, leave_log=True)
    p99_dex = torch.quantile(
        (preds - targets).abs().flatten(), float(config["target_percentile"])
    ).item()
    # loss = criterion(preds, targets).item()
    # Extract the study name without the timestamp/suffix part
    parts = study_name.split("_")
    sname = "_".join(parts[:-1]) if len(parts) > 1 else study_name

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
        return p99_dex, np.mean(inference_times)
    else:
        return p99_dex
