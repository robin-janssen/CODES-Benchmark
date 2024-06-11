import os
import h5py
import numpy as np


def check_and_load_data(dataset_name: str, mode: str, verbose: bool = True):
    """
    Check the specified dataset and load the data based on the mode (train or test).

    Args:
        dataset_name (str): The name of the dataset.
        mode (str): The mode of data to load ('train' or 'test').
        verbose (bool): Whether to print information about the loaded data.

    Returns:
        tuple: Loaded data and timesteps.

    Raises:
        Exception: If the dataset or required data is missing or if the data shape is incorrect.
    """
    data_dir = "data"
    dataset_name_lower = dataset_name.lower()

    # Check if dataset exists
    dataset_folders = [
        name.lower()
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ]
    if dataset_name_lower not in dataset_folders:
        raise Exception(
            f"Dataset '{dataset_name}' not found. Available datasets: {', '.join(dataset_folders)}"
        )

    dataset_path = os.path.join(data_dir, dataset_name_lower)
    data_file_path = os.path.join(dataset_path, "data.hdf5")

    # Check if data.hdf5 file exists
    if not os.path.exists(data_file_path):
        raise Exception(f"data.hdf5 file not found in {dataset_path}")

    with h5py.File(data_file_path, "r") as f:
        # Check if mode data exists
        if mode not in f:
            raise Exception(
                f"'{mode}' data not found in {data_file_path}. Available data: {', '.join(list(f.keys()))}"
            )

        # Load data
        data = f[mode][:]

        # Check data shape
        if data.ndim != 3:
            raise Exception(
                f"{mode} data does not have the required shape (n_samples, n_timesteps, n_chemicals)."
            )

        n_samples, n_timesteps, n_chemicals = data.shape
        if verbose:
            print(
                f"{mode.capitalize()} data loaded: {n_samples} samples, {n_timesteps} timesteps, {n_chemicals} chemicals."
            )

        # Load or generate timesteps
        if "timesteps" in f:
            timesteps = f["timesteps"][:]
            if verbose:
                print(
                    f"Timesteps loaded from data.hdf5: {timesteps.shape[0]} timesteps."
                )
        else:
            timesteps = np.linspace(0, n_timesteps - 1, n_timesteps)
            if verbose:
                print(
                    f"Timesteps not found in data.hdf5. Generated integer timesteps: {n_timesteps} timesteps."
                )

        if "n_train_samples" in f.attrs:
            n_train_samples = f.attrs["n_train_samples"]
            if verbose:
                print(f"Number of training samples: {n_train_samples}")
        else:
            n_train_samples = None
            if verbose:
                print("Number of training samples not found in metadata.")

    return data, timesteps, n_train_samples


def create_hdf5_dataset(
    train_data: np.ndarray,
    test_data: np.ndarray,
    dataset_name: str,
    data_dir: str = "data",
    timesteps: np.ndarray = None,
):
    """
    Create an HDF5 file for a dataset with train and test data, and optionally timesteps.
    Additionally, store metadata about the dataset.

    Args:
        train_data (np.ndarray): The training data array of shape (n_samples, n_timesteps, n_chemicals).
        test_data (np.ndarray): The test data array of shape (n_samples, n_timesteps, n_chemicals).
        dataset_name (str): The name of the dataset.
        data_dir (str): The directory to save the dataset in.
        timesteps (np.ndarray, optional): The timesteps array. If None, integer timesteps will be generated.

    Raises:
        ValueError: If the data does not have the required shape.
    """
    # Check data shapes
    if train_data.ndim != 3:
        raise ValueError(
            "Train data does not have the required shape (n_samples, n_timesteps, n_chemicals)."
        )
    if test_data.ndim != 3:
        raise ValueError(
            "Test data does not have the required shape (n_samples, n_timesteps, n_chemicals)."
        )

    # Generate timesteps if not provided
    if timesteps is None:
        print("Timesteps not provided and will not be saved.")
    else:
        # Ensure timesteps have the correct shape
        if timesteps.ndim != 1 or timesteps.shape[0] != train_data.shape[1]:
            raise ValueError(
                "Timesteps must be a 1D array with length equal to the number of timesteps in the data."
            )

    # Create dataset directory if it doesn't exist
    dataset_dir = os.path.join(data_dir, dataset_name.lower())
    os.makedirs(dataset_dir, exist_ok=True)

    # Create HDF5 file and save data and metadata
    data_file_path = os.path.join(dataset_dir, "data.hdf5")
    with h5py.File(data_file_path, "w") as f:
        f.create_dataset("train", data=train_data)
        f.create_dataset("test", data=test_data)
        if timesteps is not None:
            f.create_dataset("timesteps", data=timesteps)

        # Store metadata
        f.attrs["n_train_samples"] = train_data.shape[0]
        f.attrs["n_chemicals"] = train_data.shape[2]

    print(f"HDF5 dataset created at {data_file_path}")


def get_data_subset(full_train_data, full_test_data, osu_timesteps, mode, metric):
    """
    Get the appropriate data subset based on the mode and metric.

    Args:
        full_train_data (np.ndarray): The full training data.
        full_test_data (np.ndarray): The full test data.
        osu_timesteps (np.ndarray): The timesteps.
        mode (str): The benchmark mode (e.g., "accuracy", "interpolation", "extrapolation", "sparse", "UQ").
        metric (str): The specific metric for the mode (e.g., interval, cutoff, factor).

    Returns:
        tuple: The training data, test data, and timesteps subset.
    """
    if mode == "interpolation":
        interval = int(metric)
        train_data = full_train_data[:, ::interval]
        test_data = full_test_data[:, ::interval]
        timesteps = osu_timesteps[::interval]
    elif mode == "extrapolation":
        cutoff = int(metric)
        train_data = full_train_data[:, :cutoff]
        test_data = full_test_data[:, :cutoff]
        timesteps = osu_timesteps[:cutoff]
    elif mode == "sparse":
        factor = int(metric)
        train_data = full_train_data[::factor]
        test_data = full_test_data[::factor]
        timesteps = osu_timesteps
    else:
        train_data = full_train_data
        test_data = full_test_data
        timesteps = osu_timesteps

    return train_data, test_data, timesteps
