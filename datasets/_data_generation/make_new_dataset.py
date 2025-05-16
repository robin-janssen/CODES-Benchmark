import sys

import numpy as np

# Add the path where create_dataset is defined (adjust if needed)
sys.path.insert(1, "../..")
from codes import create_dataset

if __name__ == "__main__":
    # ------------------------------
    # Generate Dummy Data and Parameters
    # ------------------------------
    # Define dimensions for our dummy data.
    # n_samples: total number of trajectories
    # n_timesteps: number of time steps in each trajectory
    # n_quantities: number of quantities (e.g., chemical species) per time step
    n_samples = 300  # total number of trajectories
    n_timesteps = 100  # e.g., 100 time points
    n_quantities = 5  # e.g., 5 chemical species

    # Create dummy trajectory data.
    # The data is a 3D array with shape (n_samples, n_timesteps, n_quantities)
    full_dataset = np.random.rand(n_samples, n_timesteps, n_quantities)

    # Optionally, create dummy parameters for each trajectory.
    # For example, let’s assume each trajectory has an associated parameter vector with 2 values.
    n_parameters = 2
    # The parameters array is 2D with shape (n_samples, n_parameters).
    full_params = np.random.rand(n_samples, n_parameters)

    # ------------------------------
    # Shuffle and Prepare the Data
    # ------------------------------
    # Shuffle the data (and parameters) to ensure randomness.
    # This is important if you plan to later split the dataset.
    permutation = np.random.permutation(n_samples)
    full_dataset = full_dataset[permutation]
    full_params = full_params[permutation]

    # ------------------------------
    # Define Timesteps and Split Ratios
    # ------------------------------
    # Create a timesteps array. The length of this array must match n_timesteps.
    timesteps = np.linspace(0, 1, n_timesteps)

    # Define a split tuple for training, testing, and validation data.
    # These values must sum to 1. Here we use 70% for training, 20% for testing, and 10% for validation.
    split = (0.7, 0.1, 0.2)

    # ------------------------------
    # Optional: Define Labels
    # ------------------------------
    # For demonstration purposes, we leave labels as None.
    # If you have labels for each chemical/quantity, define them in a list of strings
    # with the same length as n_quantities.
    labels = None

    # ------------------------------
    # Create the Dataset with Parameters
    # ------------------------------
    # The create_dataset function now accepts a single "data" argument and an optional "params" argument.
    # When passing a single array (and corresponding parameters), a split tuple is required to generate
    # training, test, and validation subsets.
    create_dataset(
        "dummy_ode_dataset",
        data=full_dataset,
        params=full_params,  # These parameters are optional and not required for the benchmark.
        timesteps=timesteps,
        labels=labels,
        split=split,
    )

    print("Dataset created successfully!")
