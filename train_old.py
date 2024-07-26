# This is the main training script for the models. It trains all required models and saves them.
# The structure of the file should be something like this:

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np

from surrogates.surrogate_classes import surrogate_classes
from data import check_and_load_data
from utils import load_and_save_config

import torch

torch.cuda.init()


def train_and_save_model(
    mode: str,
    surrogate_name: str,
    train_data: np.ndarray,
    test_data: np.ndarray,
    timesteps: np.ndarray,
    surrogate_class,
    config,
    extra_info: str = "",
):
    """
    Train and save a model for a specific benchmark mode.

    Args:
        mode (str): The benchmark mode (e.g., "accuracy", "interpolation", "extrapolation", "sparse", "UQ").
        surrogate_name (str): The name of the surrogate model.
        train_data (np.ndarray): The training data.
        test_data (np.ndarray): The test data.
        timesteps (np.ndarray): The timesteps.
        surrogate_class: The class of the surrogate model.
        config (dict): The configuration dictionary.
        extra_info (str): Additional information to include in the model name (e.g. interval, cutoff, factor, etc.)
    """
    device = (
        config["devices"][0]
        if isinstance(config["devices"], list)
        else config["devices"]
    )

    # Instantiate the model to access its internal configuration
    model = surrogate_class(device=device)

    train_loader = model.prepare_data(
        dataset=train_data, timesteps=timesteps, shuffle=True
    )
    test_loader = model.prepare_data(
        dataset=test_data, timesteps=timesteps, shuffle=False
    )

    # Train the model
    model.fit(train_loader=train_loader, test_loader=test_loader, timesteps=timesteps)

    # Save the model (making the name lowercase and removing any underscores)

    model_name = f"{surrogate_name.lower()}_{mode}_{extra_info}".strip("_")
    # Remove any double underscores
    model_name = model_name.replace("__", "_")
    model.save(
        model_name=model_name,
        training_id=config["training_id"],
        dataset_name=config["dataset"]["name"],
    )


def train_surrogate(config, surrogate_class, surrogate_name):
    """
    Train and save models for different purposes based on the config settings.

    Args:
        config (dict): The configuration dictionary.
        surrogate_class: The class of the surrogate model to train.
        surrogate_name (str): The name of the surrogate model.
    """

    # Load data
    full_train_data, full_test_data, _, timesteps, _ = check_and_load_data(
        config["dataset"]["name"], verbose=False, log=config["log"]
    )

    # Just for testing purposes
    # full_train_data = full_train_data[:20]
    # full_test_data = full_test_data[:20]

    print(f"Loaded data with shape: {full_train_data.shape}/{full_test_data.shape}")

    # Main model for timing and accuracy
    if config["accuracy"]:
        print("Training main model...")
        train_and_save_model(
            "main",
            surrogate_name,
            full_train_data,
            full_test_data,
            timesteps,
            surrogate_class,
            config,
        )

    # Models for interpolation testing
    if config["interpolation"]["enabled"]:
        print("Training interpolation models...")
        for interval in config["interpolation"]["intervals"]:
            train_data = full_train_data[:, ::interval]
            test_data = full_test_data[:, ::interval]
            train_and_save_model(
                "interpolation",
                surrogate_name,
                train_data,
                test_data,
                timesteps[::interval],
                surrogate_class,
                config,
                extra_info=str(interval),
            )

    # Models for extrapolation testing
    if config["extrapolation"]["enabled"]:
        print("Training extrapolation models...")
        for cutoff in config["extrapolation"]["cutoffs"]:
            train_data = full_train_data[:, :cutoff]
            test_data = full_test_data[:, :cutoff]
            train_and_save_model(
                "extrapolation",
                surrogate_name,
                train_data,
                test_data,
                timesteps[:cutoff],
                surrogate_class,
                config,
                extra_info=str(cutoff),
            )

    # Sparse data performance testing
    if config["sparse"]["enabled"]:
        print("Training sparse models...")
        for factor in config["sparse"]["factors"]:
            train_data = full_train_data[::factor]
            test_data = full_test_data[::factor]
            train_and_save_model(
                "sparse",
                surrogate_name,
                train_data,
                test_data,
                timesteps,
                surrogate_class,
                config,
                extra_info=str(factor),
            )

    # UQ using deep ensemble
    if config["UQ"]["enabled"]:
        print("Training UQ models...")
        n_models = config["UQ"]["n_models"]
        for i in range(n_models - 1):
            # Shuffle the training data
            shuffled_indices = np.random.permutation(full_train_data.shape[0])
            shuffled_train_data = full_train_data[shuffled_indices]

            train_and_save_model(
                "UQ",
                surrogate_name,
                shuffled_train_data,
                full_test_data,
                timesteps,
                surrogate_class,
                config,
                extra_info=str(i + 1),
            )


# 3. Define the main function


def main():
    # Load configuration from YAML
    config = load_and_save_config()
    for surrogate_name in config["surrogates"]:
        if surrogate_name in surrogate_classes:
            surrogate_class = surrogate_classes[surrogate_name]
            train_surrogate(config, surrogate_class, surrogate_name)
        else:
            print(f"Surrogate {surrogate_name} not recognized. Skipping.")


# 4. Run the main function if executed as a script

if __name__ == "__main__":
    main()