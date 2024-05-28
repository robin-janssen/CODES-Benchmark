import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import os


def save_plot(plt, filename: str, conf: dict, surr_name: str) -> None:
    """
    Save the plot to a file, creating necessary directories if they don't exist.

    Args:
        plt (matplotlib.pyplot): The plot object to save.
        filename (str): The desired filename for the plot.
        conf (dict): The configuration dictionary.
        surr_name (str): The name of the surrogate model.

    Raises:
        ValueError: If the configuration dictionary does not contain the required keys.
    """
    if "training_ID" not in conf:
        raise ValueError("Configuration dictionary must contain 'training_ID'.")

    training_id = conf["training_ID"]
    plot_dir = os.path.join("plots", training_id, surr_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    filepath = save_plot_counter(filename, plot_dir)
    plt.savefig(filepath)
    print(f"Plot saved as: {filepath}")


def save_plot_counter(filename: str, directory: str) -> str:
    """
    Save a plot with an incremented filename if a file with the same name already exists.

    Args:
        filename (str): The desired filename for the plot.
        directory (str): The directory to save the plot in.

    Returns:
        str: The full path to the saved plot.
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    filepath = os.path.join(directory, filename)

    while os.path.exists(filepath):
        filepath = os.path.join(directory, f"{base}_{counter}{ext}")
        counter += 1

    return filepath


def plot_relative_errors_over_time(
    surr_name: str,
    conf: dict,
    relative_errors: np.ndarray,
    title: str,
    save: bool = False,
) -> None:
    """
    Plot the mean and median relative errors over time with shaded regions for
    the 50th, 90th, and 99th percentiles.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        relative_errors (np.ndarray): The relative errors of the model.
        title (str): The title of the plot.
        save (bool): Whether to save the plot.
    """
    # Calculate the mean, median, and percentiles across all samples and chemicals
    mean_errors = np.mean(relative_errors, axis=(0, 2))
    median_errors = np.median(relative_errors, axis=(0, 2))
    p50_upper = np.percentile(relative_errors, 75, axis=(0, 2))
    p50_lower = np.percentile(relative_errors, 25, axis=(0, 2))
    p90_upper = np.percentile(relative_errors, 95, axis=(0, 2))
    p90_lower = np.percentile(relative_errors, 5, axis=(0, 2))
    p99_upper = np.percentile(relative_errors, 99.5, axis=(0, 2))
    p99_lower = np.percentile(relative_errors, 0.5, axis=(0, 2))

    timesteps = np.arange(relative_errors.shape[1])

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_errors, label="Mean Error", color="blue")
    plt.plot(timesteps, median_errors, label="Median Error", color="red")

    # Shading areas
    plt.fill_between(
        timesteps,
        p50_lower,
        p50_upper,
        color="grey",
        alpha=0.45,
        label="50th Percentile",
    )
    plt.fill_between(
        timesteps,
        p90_lower,
        p90_upper,
        color="grey",
        alpha=0.4,
        label="90th Percentile",
    )
    plt.fill_between(
        timesteps,
        p99_lower,
        p99_upper,
        color="grey",
        alpha=0.15,
        label="99th Percentile",
    )

    plt.yscale("log")
    plt.xlabel("Timestep")
    plt.ylabel("Relative Error (Log Scale)")
    plt.title(title)
    plt.legend()

    if save and conf:
        save_plot(plt, "relative_errors.png", conf, surr_name)

    # plt.show()

    plt.close()


def plot_dynamic_correlation(
    surr_name: str,
    conf: dict,
    gradients: np.ndarray,
    errors: np.ndarray,
    save: bool = False,
):
    """
    Plot the correlation between the gradients of the data and the prediction errors.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        gradients (np.ndarray): The gradients of the data.
        errors (np.ndarray): The prediction errors.
        save (bool): Whether to save the plot.
    """
    # Flatten the arrays for correlation plot
    gradients_flat = gradients.flatten()
    errors_flat = errors.flatten()

    # Scatter plot of gradients vs. errors
    plt.figure(figsize=(10, 6))
    plt.scatter(gradients_flat, errors_flat, alpha=0.5, s=5)
    plt.xlabel("Gradient of Data")
    plt.ylabel("Prediction Error")
    plt.title("Correlation between Gradients and Prediction Errors")

    # Save the plot
    if save and conf:
        save_plot(plt, "dynamic_correlation.png", conf, surr_name)

    # plt.show()

    plt.close()


def plot_generalization_errors(
    surr_name: str,
    conf: dict,
    metrics: np.array,
    model_errors: np.array,
    interpolate: bool = True,
    save: bool = False,
) -> None:
    """
    Plot the interpolation or extrapolation errors of a model.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        metrics (np.array): The interpolation intervals or extrapolation cutoffs.
        model_errors (np.array): The mean absolute errors of the model.
        interpolate (bool): Whether to plot interpolation errors. If False, extrapolation errors are plotted.
        save (bool): Whether to save the plot.

    Returns:
        None
    """
    xlabel = "Interpolation Interval" if interpolate else "Extrapolation Cutoff"
    title = "Interpolation Errors" if interpolate else "Extrapolation Errors"
    filename = "interpolation_errors.png" if interpolate else "extrapolation_errors.png"
    plt.scatter(metrics, model_errors)
    plt.xlabel(xlabel)
    plt.ylabel("Mean Absolute Error")
    plt.yscale("log")
    plt.title(title)

    if save and conf:
        save_plot(plt, filename, conf, surr_name)

    # plt.show()

    plt.close()


def plot_sparse_errors(
    surr_name: str,
    conf: dict,
    n_train_samples: np.ndarray,
    model_errors: np.ndarray,
    title: Optional[str] = None,
    save: bool = False,
) -> None:
    """
    Plot the sparse training errors of a model.

    Args:
        surr_name: The name of the surrogate model.
        conf: The configuration dictionary.
        n_train_samples: Numpy array containing the number of training samples.
        model_errors: Numpy array containing the model errors.
        title: Optional title for the plot.
        save: Whether to save the plot as a file.

    Returns:
        None
    """
    plt.scatter(n_train_samples, model_errors)
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Mean Absolute Error")
    plt.yscale("log")
    if title is None:
        title = "Sparse Training Errors"
    plt.title(title)

    if save and conf:
        save_plot(plt, "sparse_errors.png", conf, surr_name)

    plt.show()
