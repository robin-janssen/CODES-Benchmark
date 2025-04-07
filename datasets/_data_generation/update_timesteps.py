import numpy as np

from codes.utils.data_utils import check_and_load_data, create_dataset


def main():
    """
    Main function to create the timesteps in an existing HDF5 dataset.
    It loads the dataset, prints the current timesteps, and updates the timesteps.

    The integration was performed in log‐space, but for interpretability we wish to store the
    corresponding linear (real) time (in seconds) for each datapoint.

    Args:
        args (Namespace): The command line arguments.
    """
    # Check if the data file exists and load the data
    train_data, test_data, val_data, timesteps, n_train_samples, data_params, labels = (
        check_and_load_data("osutest", log=False, normalisation_mode="disable")
    )

    # Print the current timesteps (likely in log-space)
    print(f"Current timesteps: {timesteps}")

    # Define time parameters in seconds
    spy = 60 * 60 * 24 * 365  # seconds per year
    dt_total = 1e8 * spy  # total integration time in seconds
    tmin = spy * 1e-6  # minimum time (in seconds)
    tsteps = len(timesteps)  # expected number of timesteps (e.g., 101)

    # Create timesteps in log-space (as used during integration)
    log_timesteps = np.linspace(np.log10(tmin), np.log10(dt_total), tsteps)
    # Exponentiate to convert back to linear time (in seconds)
    new_timesteps = 10**log_timesteps

    # Update the timesteps variable with linear time values
    timesteps = new_timesteps

    # Print the new (linear) timesteps
    print(f"New timesteps (in seconds): {timesteps}")

    # Create the dataset with the new timesteps
    create_dataset(
        name="osutest2",
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        timesteps=timesteps,
        labels=list(labels),
    )


if __name__ == "__main__":
    main()
