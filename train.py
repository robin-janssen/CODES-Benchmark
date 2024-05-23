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
from surrogates.DeepONet.deeponet import MultiONet  # Import your model class here

# Load configuration from YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Define surrogate classes
surrogate_classes = {
    "DeepONet": MultiONet,
    # Add other surrogate classes here
}

# 2. Define helper functions


def train_and_save_model(
    model_name, train_data, test_data, timesteps, surrogate_class, training_id
):
    # Instantiate the model to access its internal configuration
    model = surrogate_class()
    batch_size = model.config.batch_size

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, batch_size=batch_size, shuffle=True
    )
    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, batch_size=batch_size, shuffle=False
    )

    model.fit(dataloader_train, dataloader_test)

    model.save(
        model_name=model_name,
        subfolder="models/DeepONet/trained",
        unique_id=training_id,
    )


def train_model(config, surrogate_class):
    """
    Train and save models for different purposes based on the config settings.

    Args:
        config (dict): The configuration dictionary.
        surrogate_class: The class of the surrogate model to train.
    """

    # Load data
    data_path = "data/osu_data"
    full_train_data = np.load(os.path.join(data_path, "train_data.npy"))
    full_test_data = np.load(os.path.join(data_path, "test_data.npy"))
    osu_timesteps = np.linspace(0, 99, 100)

    # Just for testing purposes
    # full_train_data = full_train_data[:32]
    # full_test_data = full_test_data[:32]

    print(f"Loaded data with shape: {full_train_data.shape}/{full_test_data.shape}")

    training_id = config["training_ID"]

    # Main model for timing and accuracy
    if config["accuracy"]:
        train_and_save_model(
            "main_deeponet.pth",
            full_train_data,
            full_test_data,
            osu_timesteps,
            surrogate_class,
            training_id,
        )

    # Models for interpolation testing
    if config["interpolation"]["enabled"]:
        for interval in config["interpolation"]["intervals"]:
            train_data = full_train_data[:, ::interval]
            test_data = full_test_data[:, ::interval]
            train_and_save_model(
                f"deeponet_interpolation_{interval}.pth",
                train_data,
                test_data,
                osu_timesteps[::interval],
                surrogate_class,
                training_id,
            )

    # Models for extrapolation testing
    if config["extrapolation"]["enabled"]:
        for cutoff in config["extrapolation"]["cutoffs"]:
            train_data = full_train_data[:, :cutoff]
            test_data = full_test_data[:, :cutoff]
            train_and_save_model(
                f"deeponet_extrapolation_{cutoff}.pth",
                train_data,
                test_data,
                osu_timesteps[:cutoff],
                surrogate_class,
                training_id,
            )

    # Sparse data performance testing
    if config["sparse"]["enabled"]:
        for factor in config["sparse"]["factors"]:
            train_data = full_train_data[::factor]
            test_data = full_test_data[::factor]
            train_and_save_model(
                f"deeponet_sparse_{factor}.pth",
                train_data,
                test_data,
                osu_timesteps[::factor],
                surrogate_class,
                training_id,
            )

    # UQ using deep ensemble
    if config["UQ"]["enabled"]:
        n_models = config["UQ"]["n_models"]
        for i in range(n_models):
            train_and_save_model(
                f"deeponet_ensemble_{i}.pth",
                full_train_data,
                full_test_data,
                osu_timesteps,
                surrogate_class,
                training_id,
            )


# 3. Define the main function


def main():
    for surrogate_name in config["surrogates"]:
        if surrogate_name in surrogate_classes:
            surrogate_class = surrogate_classes[surrogate_name]
            train_model(config, surrogate_class)
        else:
            print(f"Surrogate {surrogate_name} not recognized. Skipping.")


# 4. Run the main function if executed as a script

if __name__ == "__main__":
    main()
