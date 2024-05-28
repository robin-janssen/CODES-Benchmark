# This is the main training script for the models. It trains all required models and saves them.
# The structure of the file should be something like this:

# 1. Import the necessary libraries
# 2. Define helper functions
# 3. Define the main training logic
# 4. Run the main function if executed as a script

# 1. Import the necessary libraries

import yaml

from surrogates.surrogate_classes import surrogate_classes
from data import check_and_load_data

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

    # Train the model
    model.fit(train_data, test_data, timesteps)

    # Save the model
    # Make the name all lowercase

    model_name = f"{surrogate_name.lower()}_{mode}_{extra_info}".strip("_")
    # Remove any double underscores
    model_name = model_name.replace("__", "_")
    model.save(
        model_name=model_name,
        subfolder=f"trained/{surrogate_name}",
        unique_id=training_id,
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
    full_train_data, osu_timesteps, _ = check_and_load_data(config["dataset"], "train")
    full_test_data, _, _ = check_and_load_data(config["dataset"], "test")

    # Just for testing purposes
    full_train_data = full_train_data[:32]
    full_test_data = full_test_data[:32]

    print(f"Loaded data with shape: {full_train_data.shape}/{full_test_data.shape}")

    training_id = config["training_ID"]

    # Main model for timing and accuracy
    if config["accuracy"]:
        print("Training main model...")
        train_and_save_model(
            "main",
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
        for i in range(n_models - 1):
            train_and_save_model(
                "UQ",
                surrogate_name,
                full_train_data,
                full_test_data,
                osu_timesteps,
                surrogate_class,
                training_id,
                extra_info=str(i + 1),
            )


# 3. Define the main function


def main():
    for surrogate_name in config["surrogates"]:
        if surrogate_name in surrogate_classes:
            surrogate_class = surrogate_classes[surrogate_name]
            train_surrogate(config, surrogate_class, surrogate_name)
        else:
            print(f"Surrogate {surrogate_name} not recognized. Skipping.")


# 4. Run the main function if executed as a script

if __name__ == "__main__":
    main()
