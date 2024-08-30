import os

import yaml
from torch import nn

import optuna
from benchmark.bench_utils import get_surrogate
from data import check_and_load_data
from data.data_utils import download_data
from utils import make_description, set_random_seeds

# Configuration dictionary with optuna_params nested
config = {
    "surrogate": {
        "name": "MultiONet",
    },
    "dataset": {
        "name": "osu2008",
        "log10_transform": False,
        "normalise": "minmax",
    },
    "device": "cuda:0",
    "seed": 42,
    "batch_size": 1024,
    "epochs": 500,
    "study_name": "multionet_osu",
    "optuna_params": {
        "branch_hidden_layers": {"type": "int", "low": 2, "high": 8},
        "trunk_hidden_layers": {"type": "int", "low": 2, "high": 8},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "output_factor": {"type": "int", "low": 1, "high": 30},
        "activation_function": {
            "type": "categorical",
            "choices": ["relu", "tanh", "sigmoid"],
        },
    },
}


def save_optuna_config(config, folder_path="optuna/studies/"):
    """
    Function to save the Optuna configuration to a YAML file.

    Args:
        config (dict): The configuration dictionary to save.
        folder_path (str): The folder path where the YAML file will be saved.
    """
    # Create directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Define the file path
    file_path = os.path.join(folder_path, f"{config['study_name']}_config.yaml")

    # Save the configuration as a YAML file
    with open(file_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


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


def objective(trial):
    return training_run(trial, config)


def training_run(trial, config):
    # Load data
    download_data(config["dataset"]["name"])
    train_data, test_data, val_data, timesteps, _, data_params, _ = check_and_load_data(
        config["dataset"]["name"],
        verbose=False,
        log=config["dataset"]["log10_transform"],
        normalisation_mode=config["dataset"]["normalise"],
    )

    set_random_seeds(config["seed"])

    surr_name = config["surrogate"]["name"]

    # Generate suggested parameters using the make_optuna_params function
    suggested_params = make_optuna_params(trial, config["optuna_params"])

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
        subfolder="optuna",
        data_params=data_params,
    )

    return loss


def run(config):
    # Save the configuration before starting the study
    save_optuna_config(config)
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    study = optuna.create_study(
        study_name=config["study_name"],
        direction="minimize",
        storage=f"sqlite:///optuna/studies/{config['study_name']}.db",
        sampler=sampler,
        load_if_exists=True,
    )
    optuna_objective = objective
    study.optimize(optuna_objective, n_trials=200)


config_2 = {
    "surrogate": {
        "name": "MultiONet",
    },
    "dataset": {
        "name": "osu2008",
        "log10_transform": False,
        "normalise": "minmax",
    },
    "device": "cuda:0",
    "seed": 42,
    "batch_size": 1024,
    "epochs": 500,
    "study_name": "multionet_osu",
    "optuna_params": {
        "branch_hidden_layers": {"type": "int", "low": 2, "high": 8},
        "trunk_hidden_layers": {"type": "int", "low": 2, "high": 8},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "output_factor": {"type": "int", "low": 1, "high": 30},
        # Example of a categorical parameter
        # "activation_function": {
        #     "type": "categorical",
        #     "choices": ["relu", "tanh", "sigmoid"],
        # },
    },
}


if __name__ == "__main__":
    run(config)
