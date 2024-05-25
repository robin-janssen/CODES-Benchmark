# This is the main training script for the models. It trains all required models and saves them.
# The structure of the file should be something like this:

# 1. Import the necessary libraries
# 2. Define helper functions
# 3. Define the main training logic
# 4. Run the main function if executed as a script

# 1. Import the necessary libraries

import os
import numpy as np
import yaml

from surrogates.DeepONet.dataloader import create_dataloader_chemicals
from surrogates.surrogates import surrogate_classes

# Load configuration from YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# 2. Define helper functions


def train_and_save_model(
    mode: str,
    surrogate_name: str,
    train_data,
    test_data,
    timesteps,
    surrogate_class,
    training_id,
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
        training_id (str): The unique identifier for the training run.
        extra_info (str): Additional information to include in the model name (e.g. interval, cutoff, factor, etc.)
    """
    # Instantiate the model to access its internal configuration
    model = surrogate_class()
    batch_size = model.config.batch_size

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, batch_size=batch_size, shuffle=True
    )
    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, batch_size=batch_size, shuffle=False
    )

    model.config.N_timesteps = len(timesteps)

    model.train_model(dataloader_train, dataloader_test)

    model_name = f"{mode}_{extra_info}_{surrogate_name}.pth".strip("_")
    model.save(
        model_name=model_name,
        subfolder=f"trained/{surrogate_name}",
        unique_id=training_id,
    )


def train_model(config, surrogate_class, surrogate_name):
    """
    Train and save models for different purposes based on the config settings.

    Args:
        config (dict): The configuration dictionary.
        surrogate_class: The class of the surrogate model to train.
        surrogate_name (str): The name of the surrogate model.
    """

    # Load data
    data_path = "data/osu_data"
    full_train_data = np.load(os.path.join(data_path, "train_data.npy"))
    full_test_data = np.load(os.path.join(data_path, "test_data.npy"))
    osu_timesteps = np.linspace(0, 99, 100)

    # Just for testing purposes
    full_train_data = full_train_data[:32]
    full_test_data = full_test_data[:32]

    print(f"Loaded data with shape: {full_train_data.shape}/{full_test_data.shape}")

    training_id = config["training_ID"]

    # Main model for timing and accuracy
    if config["accuracy"]:
        print("Training main model...")
        train_and_save_model(
            "accuracy",
            surrogate_name,
            full_train_data,
            full_test_data,
            osu_timesteps,
            surrogate_class,
            training_id,
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
                osu_timesteps[::interval],
                surrogate_class,
                training_id,
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
                osu_timesteps[:cutoff],
                surrogate_class,
                training_id,
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
                osu_timesteps,
                surrogate_class,
                training_id,
                extra_info=str(factor),
            )

    # UQ using deep ensemble
    if config["UQ"]["enabled"]:
        print("Training UQ models...")
        n_models = config["UQ"]["n_models"]
        for i in range(n_models):
            train_and_save_model(
                "UQ",
                surrogate_name,
                full_train_data,
                full_test_data,
                osu_timesteps,
                surrogate_class,
                training_id,
                extra_info=str(i),
            )


# 3. Define the main function


def main():
    for surrogate_name in config["surrogates"]:
        if surrogate_name in surrogate_classes:
            surrogate_class = surrogate_classes[surrogate_name]
            train_model(config, surrogate_class, surrogate_name)
        else:
            print(f"Surrogate {surrogate_name} not recognized. Skipping.")


# 4. Run the main function if executed as a script

if __name__ == "__main__":
    main()
