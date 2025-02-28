import os
import urllib.request
from math import isclose

import h5py
import numpy as np
import yaml
from tqdm import tqdm


class DatasetError(Exception):
    """
    Error for missing data or dataset or if the data shape is incorrect.
    """

    pass


def check_and_load_data(
    dataset_name: str,
    verbose: bool = True,
    log: bool = True,
    normalisation_mode: str = "standardise",
    tolerance: float | None = None,
):
    """
    Check the specified dataset and load the data based on the mode (train or test).

    Args:
        dataset_name (str): The name of the dataset.
        verbose (bool): Whether to print information about the loaded data.
        log (bool): Whether to log-transform the data (log10).
        normalisation_mode (str): The normalization mode, either "disable", "minmax", or "standardise".
        tolerance (float, optional): The tolerance value for log-transformation.
            Values below this will be set to the tolerance value. Pass None to disable.

    Returns:
        tuple: Loaded data and timesteps.

    Raises:
        DatasetError: If the dataset or required data is missing or if the data shape is incorrect.
    """
    data_dir = "datasets"
    dataset_name_lower = dataset_name.lower()

    # Check if dataset exists
    dataset_folders = [
        name.lower()
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ]
    if dataset_name_lower not in dataset_folders:
        raise DatasetError(
            f"Dataset '{dataset_name}' not found. Available datasets: {', '.join(dataset_folders)}"
        )

    dataset_path = os.path.join(data_dir, dataset_name_lower)
    data_file_path = os.path.join(dataset_path, "data.hdf5")

    # Check if data.hdf5 file exists
    if not os.path.exists(data_file_path):
        raise DatasetError(f"data.hdf5 file not found in {dataset_path}")

    with h5py.File(data_file_path, "r") as f:

        for mode in ("train", "test", "val"):
            if mode not in f:
                raise DatasetError(
                    f"'{mode}' data not found in {data_file_path}. Available data: {', '.join(list(f.keys()))}"
                )

        # Load data
        train_data = np.asarray(f["train"], dtype=np.float32)
        test_data = np.asarray(f["test"], dtype=np.float32)
        val_data = np.asarray(f["val"], dtype=np.float32)

        if tolerance is not None:
            tolerance = np.float32(tolerance)
            train_data = np.where(train_data < tolerance, tolerance, train_data)
            test_data = np.where(test_data < tolerance, tolerance, test_data)
            val_data = np.where(val_data < tolerance, tolerance, val_data)

        # Log transformation
        if log:
            try:
                train_data = np.log10(train_data)
                test_data = np.log10(test_data)
                val_data = np.log10(val_data)
                if verbose:
                    print("Data log-transformed.")
            except ValueError:
                print(
                    "Data contains non-positive values and cannot be log-transformed."
                )

        # Normalization
        if normalisation_mode not in ["disable", "minmax", "standardise"]:
            raise ValueError(
                "Normalization mode must be either 'disable', 'minmax' or 'standardise'"
            )
        if normalisation_mode != "disable":
            data_params, train_data, test_data, val_data = normalize_data(
                train_data, test_data, val_data, mode=normalisation_mode
            )
            if verbose:
                print(f"Data normalized (mode: {normalisation_mode}).")
        else:  # No normalization
            data_params = {"mode": "disable"}

        data_params["log10_transform"] = True if log else False

        # Check data shape
        for data in (train_data, test_data, val_data):
            if data.ndim != 3:
                raise DatasetError(
                    "Data does not have the required shape (n_samples, n_timesteps, n_chemicals)."
                )

        _, n_timesteps, n_chemicals = train_data.shape
        n_samples = train_data.shape[0] + test_data.shape[0] + val_data.shape[0]
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
            timesteps = np.linspace(0, 1, n_timesteps)
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

        data_params["dataset_name"] = dataset_name

        if "labels" in f.attrs:
            labels = f.attrs["labels"]
            if not isinstance(labels, np.ndarray):
                raise TypeError("Labels must be an array of strings.")
            if len(labels) != n_chemicals:
                raise ValueError(
                    "The number of labels must match the number of chemicals."
                )
        else:
            labels = None
            if verbose:
                print("Labels not found in metadata.")

    return (
        train_data,
        test_data,
        val_data,
        timesteps,
        n_train_samples,
        data_params,
        labels,
    )


def normalize_data(
    train_data: np.ndarray,
    test_data: np.ndarray | None = None,
    val_data: np.ndarray | None = None,
    mode: str = "standardise",
) -> tuple:
    """
    Normalize the data based on the training data statistics.

    Args:
        train_data (np.ndarray): Training data array.
        test_data (np.ndarray, optional): Test data array.
        val_data (np.ndarray, optional): Validation data array.
        mode (str): Normalization mode, either "minmax" or "standardise".

    Returns:
        tuple: Normalized training data, test data, and validation data.
    """
    if mode not in ["minmax", "standardise"]:
        raise ValueError("Mode must be either 'minmax' or 'standardise'")

    if mode == "minmax":
        # Compute min and max on the training data
        data_min = np.min(train_data)
        data_max = np.max(train_data)

        data_params = {"min": float(data_min), "max": float(data_max), "mode": mode}

        # Normalize the training data
        train_data_norm = 2 * (train_data - data_min) / (data_max - data_min) - 1

        if test_data is not None:
            test_data_norm = 2 * (test_data - data_min) / (data_max - data_min) - 1
        else:
            test_data_norm = None

        if val_data is not None:
            val_data_norm = 2 * (val_data - data_min) / (data_max - data_min) - 1
        else:
            val_data_norm = None

    elif mode == "standardise":
        # Compute mean and std on the training data
        mean = np.mean(train_data)
        std = np.std(train_data)

        data_params = {"mean": float(mean), "std": float(std), "mode": mode}

        # Standardize the training data
        train_data_norm = (train_data - mean) / std

        if test_data is not None:
            test_data_norm = (test_data - mean) / std
        else:
            test_data_norm = None

        if val_data is not None:
            val_data_norm = (val_data - mean) / std
        else:
            val_data_norm = None

    return data_params, train_data_norm, test_data_norm, val_data_norm


def create_hdf5_dataset(
    train_data: np.ndarray,
    test_data: np.ndarray,
    val_data: np.ndarray,
    dataset_name: str,
    data_dir: str = "datasets",
    timesteps: np.ndarray | None = None,
    labels: list[str] | None = None,
):
    """
    Create an HDF5 file for a dataset with train and test data, and optionally timesteps.
    Additionally, store metadata about the dataset.

    Args:
        train_data (np.ndarray): The training data array of shape (n_samples, n_timesteps, n_chemicals).
        test_data (np.ndarray): The test data array of shape (n_samples, n_timesteps, n_chemicals).
        val_data (np.ndarray): The validation data array of shape (n_samples, n_timesteps, n_chemicals).
        dataset_name (str): The name of the dataset.
        data_dir (str): The directory to save the dataset in.
        timesteps (np.ndarray, optional): The timesteps array. If None, integer timesteps will be generated.
        labels (list[str], optional): The labels for the chemicals.
    """

    # Create dataset directory if it doesn't exist
    dataset_dir = os.path.join(data_dir, dataset_name.lower())
    os.makedirs(dataset_dir, exist_ok=True)

    # Create HDF5 file and save data and metadata
    data_file_path = os.path.join(dataset_dir, "data.hdf5")
    with h5py.File(data_file_path, "w") as f:
        f.create_dataset("train", data=train_data)
        f.create_dataset("test", data=test_data)
        f.create_dataset("val", data=val_data)
        if timesteps is not None:
            f.create_dataset("timesteps", data=timesteps)

        # Store metadata
        f.attrs["n_train_samples"] = train_data.shape[0]
        f.attrs["n_test_samples"] = test_data.shape[0]
        f.attrs["n_val_samples"] = val_data.shape[0]
        f.attrs["n_timesteps"] = train_data.shape[1]
        f.attrs["n_chemicals"] = train_data.shape[2]
        f.attrs["labels"] = labels


def get_data_subset(
    full_train_data: np.ndarray,
    full_test_data: np.ndarray,
    timesteps: np.ndarray,
    mode: str,
    metric: int,
    subset_factor: int = 1,
):
    """
    Get the appropriate data subset based on the mode and metric.

    Args:
        full_train_data (np.ndarray): The full training data.
        full_test_data (np.ndarray): The full test data.
        timesteps (np.ndarray): The timesteps.
        mode (str): The benchmark mode (e.g., "accuracy", "interpolation", "extrapolation", "sparse", "UQ").
        metric (int): The specific metric for the mode (e.g., interval, cutoff, factor, batch size).
        subset_factor (int): The factor to subset the data by. Default is 1 (use full train and test data).

    Returns:
        tuple: The training data, test data, and timesteps subset.
    """
    # First select the appropriate data subset
    train_data = full_train_data[::subset_factor]
    test_data = full_test_data[::subset_factor]

    if mode == "interpolation":
        interval = metric
        train_data = train_data[:, ::interval]
        test_data = test_data[:, ::interval]
        timesteps = timesteps[::interval]
    elif mode == "extrapolation":
        cutoff = metric
        train_data = train_data[:, :cutoff]
        test_data = test_data[:, :cutoff]
        timesteps = timesteps[:cutoff]
    elif mode == "sparse":
        factor = metric
        train_data = train_data[::factor]
        test_data = test_data[::factor]
    elif mode == "batch_size":
        factor = 4
        train_data = train_data[::factor]
        test_data = test_data[::factor]
    else:
        train_data = train_data
        test_data = test_data

    return train_data, test_data, timesteps


def create_dataset(
    name: str,
    train_data: np.ndarray,
    test_data: np.ndarray | None = None,
    val_data: np.ndarray | None = None,
    split: tuple[float, float, float] | None = None,
    timesteps: np.ndarray | None = None,
    labels: list[str] | None = None,
):
    """
    Creates a new dataset in the data directory with train, test and validation data.

    If only train_data is provided, it will be split into train, test and validation sets.
    If split is not provided in this case, a default split of (0.7, 0.1, 0.2) is used.

    Args:
        name (str): The name of the dataset.
        train_data (np.ndarray): The training data.
        test_data (np.ndarray, optional): The test data.
        val_data (np.ndarray, optional): The validation data.
        split (tuple(float, float, float), optional): If test_data and val_data are not provided,
            train_data will be split into training, test and validation data according to these ratios.
        timesteps (np.ndarray, optional): The timesteps array.
        labels (list[str], optional): The labels for the chemicals.

    Raises:
        FileExistsError: If the dataset already exists.
        TypeError: If the train_data (or test_data/val_data when provided) is not a numpy array.
        ValueError: If the provided data do not have the correct shape or if only one of test_data/val_data is provided.
    """
    base_dir = "datasets"
    dataset_dir = os.path.join(base_dir, name)

    if os.path.exists(dataset_dir):
        raise FileExistsError(f"Dataset '{name}' already exists.")

    # Verify train_data is a numpy array and has the correct dimensions.
    if not isinstance(train_data, np.ndarray):
        raise TypeError("train_data must be a numpy array.")
    if train_data.ndim != 3:
        raise ValueError(
            "train_data must have shape (n_samples, n_timesteps, n_chemicals)."
        )

    # Check consistency if test_data or val_data are provided.
    if (test_data is None) ^ (val_data is None):
        raise ValueError(
            "Either provide only train_data or provide all of train_data, test_data and val_data."
        )

    # If test_data and val_data are provided, perform type and shape checks.
    if test_data is not None and val_data is not None:
        if not isinstance(test_data, np.ndarray):
            raise TypeError("test_data must be a numpy array.")
        if not isinstance(val_data, np.ndarray):
            raise TypeError("val_data must be a numpy array.")
        if train_data.shape[1:] != test_data.shape[1:]:
            raise ValueError(
                "train_data and test_data must have the same number of timesteps and chemicals."
            )
        if train_data.shape[1:] != val_data.shape[1:]:
            raise ValueError(
                "train_data and val_data must have the same number of timesteps and chemicals."
            )
        np.random.shuffle(test_data)
        np.random.shuffle(val_data)
    else:
        # Only train_data is provided. Use split to partition the data.
        if split is None:
            split = (0.7, 0.1, 0.2)  # default split
        else:
            if not isinstance(split, (tuple, list)):
                raise TypeError("split must be a tuple or list of floats.")
            if len(split) != 3:
                raise ValueError(
                    "split must contain three values: train, test, and validation split."
                )
            if not all(isinstance(value, float) for value in split):
                raise TypeError("split values must be floats.")
            if not all(0 <= value <= 1 for value in split):
                raise ValueError("split values must be between 0 and 1.")
            if not isclose(sum(split), 1, abs_tol=1e-5):
                raise ValueError("split values must sum to 1.")

    # Validate timesteps.
    if timesteps is None:
        print("Timesteps not provided and will not be saved.")
    else:
        if timesteps.ndim != 1 or timesteps.shape[0] != train_data.shape[1]:
            raise ValueError(
                "Timesteps must be a 1D array with length equal to the number of timesteps in the data."
            )

    # Shuffle the training data.
    np.random.shuffle(train_data)

    # If only train_data is provided, perform the splitting.
    if test_data is None and val_data is None:
        full_data = train_data
        n_samples = full_data.shape[0]
        n_train = int(n_samples * split[0])
        n_test = int(n_samples * split[1])

        train_data = full_data[:n_train]
        test_data = full_data[n_train : n_train + n_test]
        val_data = full_data[n_train + n_test :]

        if any(
            dim == 0
            for shape in (train_data.shape, test_data.shape, val_data.shape)
            for dim in shape
        ):
            raise ValueError(
                "Split data contains zero samples. One of the splits is too small."
            )
    else:
        # When test_data and val_data are provided, ignore the split value.
        if split is not None:
            print(
                "Warning: split values will be ignored since test_data and val_data are provided."
            )

    # Validate labels.
    if labels is not None:
        if not isinstance(labels, list):
            raise TypeError("labels must be a list of strings.")
        if len(labels) != train_data.shape[2]:
            raise ValueError("The number of labels must match the number of chemicals.")

    # Create the HDF5 dataset.
    create_hdf5_dataset(
        train_data, test_data, val_data, name, timesteps=timesteps, labels=labels
    )

    print(f"Dataset '{name}' created at {dataset_dir}")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update the progress bar.
        Args:
            b (int, optional): Number of blocks transferred so far. Default is 1.
            bsize (int, optional): Size of each block (in tqdm units). Default is 1.
            tsize (int, optional): Total size (in tqdm units). Default is None.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_data(dataset_name: str, path: str | None = None):
    """
    Download the specified dataset if it is not present, with a progress bar.
    Args:
        dataset_name (str): The name of the dataset.
        path (str, optional): The path to save the dataset. If None, the default data directory is used.
    """
    data_path = (
        os.path.abspath(f"datasets/{dataset_name.lower()}/data.hdf5")
        if path is None
        else os.path.abspath(path)
    )
    if os.path.isfile(data_path):
        print(f"Dataset '{dataset_name}' already downloaded at {data_path}.")
        return

    with open("datasets/data_sources.yaml", "r", encoding="utf-8") as file:
        data_sources = yaml.safe_load(file)

    try:
        url = data_sources[dataset_name]
    except KeyError as e:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in data_sources.yaml"
        ) from e

    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    print(f"Downloading dataset '{dataset_name}'...")
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=f"Downloading {dataset_name}"
    ) as t:
        urllib.request.urlretrieve(url, data_path, reporthook=t.update_to)
    print(f"Dataset '{dataset_name}' downloaded successfully.")
