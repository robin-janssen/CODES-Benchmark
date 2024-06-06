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
    if "training_id" not in conf:
        raise ValueError("Configuration dictionary must contain 'training_id'.")

    training_id = conf["training_id"]
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

    # plt.show()

    plt.close()


def plot_example_predictions_with_uncertainty(
    surr_name: str,
    conf: dict,
    preds_mean: np.ndarray,
    preds_std: np.ndarray,
    targets: np.ndarray,
    timesteps: np.ndarray,
    num_examples: int = 4,
    num_chemicals: int = 8,
    save: bool = False,
) -> None:
    """
    Plot example predictions with uncertainty.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        preds_mean (np.ndarray): Mean predictions from the ensemble of models.
        preds_std (np.ndarray): Standard deviation of predictions from the ensemble of models.
        targets (np.ndarray): True targets.
        timesteps (np.ndarray): Timesteps array.
        num_examples (int, optional): Number of example plots to generate.
        num_chemicals (int, optional): Number of chemicals to plot.
        save (bool, optional): Whether to save the plot as a file.
    """
    colors = plt.cm.viridis(np.linspace(0, 1, num_chemicals))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for example_idx in range(num_examples):
        ax = axs[example_idx // 2, example_idx % 2]
        for chem_idx in range(num_chemicals):
            gt = targets[example_idx, :, chem_idx]
            mean = preds_mean[example_idx, :, chem_idx]
            std = preds_std[example_idx, :, chem_idx]

            ax.plot(
                timesteps,
                gt,
                color=colors[chem_idx],
                label=f"GT Chemical {chem_idx+1}",
            )
            ax.plot(
                timesteps,
                mean,
                "--",
                color=colors[chem_idx],
                label=f"Pred Chemical {chem_idx+1}",
            )

            # Plot standard deviations as shaded areas
            for sigma_multiplier in [1, 2, 3]:  # 1, 2, and 3 standard deviations
                ax.fill_between(
                    timesteps,
                    mean - sigma_multiplier * std,
                    mean + sigma_multiplier * std,
                    color=colors[chem_idx],
                    alpha=0.5 / sigma_multiplier,
                )

    plt.legend()
    plt.tight_layout()

    if save and conf:
        save_plot(plt, "UQ_predictions.png", conf, surr_name)

    plt.close()


# def save_plot(filename: str, conf: dict, surr_name: str) -> None:
#     """
#     Save a plot to the specified directory based on the configuration.

#     Args:
#         filename (str): The name of the file to save.
#         conf (Dict): Configuration dictionary.
#         surr_name (str): The name of the surrogate model.
#     """
#     training_id = conf["training_id"]
#     plot_dir = os.path.join("plots", training_id, surr_name)
#     if not os.path.exists(plot_dir):
#         os.makedirs(plot_dir)
#     filepath = os.path.join(plot_dir, filename)
#     plt.savefig(filepath)
#     print(f"Plot saved as: {filepath}")


def plot_average_uncertainty_over_time(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    timesteps: np.ndarray,
    save: bool = False,
) -> None:
    """
    Plot the average uncertainty over time.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        preds_std (np.ndarray): Standard deviation of predictions from the ensemble of models.
        timesteps (np.ndarray): Timesteps array.
        save (bool, optional): Whether to save the plot as a file.
    """
    average_uncertainty_over_time = np.mean(preds_std, axis=(0, 2))

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, average_uncertainty_over_time, label="Average Uncertainty")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Uncertainty")
    plt.title("Average Uncertainty Over Time")
    plt.legend()

    if save and conf:
        save_plot(plt, "uncertainty_over_time.png", conf, surr_name)

    # plt.show()

    plt.close()


def plot_uncertainty_vs_errors(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    errors: np.ndarray,
    save: bool = False,
) -> None:
    """
    Plot the correlation between predictive uncertainty and prediction errors.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        preds_std (np.ndarray): Standard deviation of predictions from the ensemble of models.
        errors (np.ndarray): Prediction errors.
        save (bool, optional): Whether to save the plot as a file.
    """
    # Normalize the errors
    errors = errors / np.abs(errors).max()

    plt.figure(figsize=(10, 6))
    plt.scatter(preds_std.flatten(), errors.flatten(), alpha=0.5)
    plt.xlabel("Predictive Uncertainty")
    plt.ylabel("Prediction Error (Normalized)")
    plt.title("Correlation between Predictive Uncertainty and Prediction Errors")

    if save and conf:
        save_plot(plt, "uncertainty_vs_errors.png", conf, surr_name)

    # plt.show()

    plt.close()


def plot_surr_losses(surr_name: str, conf: dict, timesteps: np.ndarray) -> None:
    """
    Plot the training and test losses for the surrogate model.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        timesteps (np.ndarray): The timesteps array.
    """
    training_id = conf["training_id"]
    base_dir = f"trained/{training_id}/{surr_name}"

    def load_losses(model_identifier: str):
        loss_path = os.path.join(base_dir, f"{model_identifier}_losses.npz")
        with np.load(loss_path) as data:
            train_loss = data["train_loss"]
            test_loss = data["test_loss"]
        return train_loss, test_loss

    # Main model losses
    main_train_loss, main_test_loss = load_losses(f"{surr_name.lower()}_main")

    # Plot main model losses
    plot_losses(
        (main_train_loss, main_test_loss),
        ("Train Loss", "Test Loss"),
        title="Main Model Losses",
        save=True,
        conf=conf,
        surr_name=surr_name,
        mode="main",
    )

    # Interpolation losses
    if conf["interpolation"]["enabled"]:
        intervals = conf["interpolation"]["intervals"]
        interp_train_losses = [main_train_loss]
        interp_test_losses = [main_test_loss]
        for interval in intervals:
            train_loss, test_loss = load_losses(
                f"{surr_name.lower()}_interpolation_{interval}"
            )
            interp_train_losses.append(train_loss)
            interp_test_losses.append(test_loss)
        plot_losses(
            tuple(interp_train_losses),
            tuple(f"Interval {interval}" for interval in [1] + intervals),
            title="Interpolation Losses",
            save=True,
            conf=conf,
            surr_name=surr_name,
            mode="interpolation",
        )

    # Extrapolation losses
    if conf["extrapolation"]["enabled"]:
        cutoffs = conf["extrapolation"]["cutoffs"]
        extra_train_losses = [main_train_loss]
        extra_test_losses = [main_test_loss]
        for cutoff in cutoffs:
            train_loss, test_loss = load_losses(
                f"{surr_name.lower()}_extrapolation_{cutoff}"
            )
            extra_train_losses.append(train_loss)
            extra_test_losses.append(test_loss)
        plot_losses(
            tuple(extra_train_losses),
            tuple(f"Cutoff {cutoff}" for cutoff in [len(timesteps)] + cutoffs),
            title="Extrapolation Losses",
            save=True,
            conf=conf,
            surr_name=surr_name,
            mode="extrapolation",
        )

    # Sparse losses
    if conf["sparse"]["enabled"]:
        factors = conf["sparse"]["factors"]
        sparse_train_losses = [main_train_loss]
        sparse_test_losses = [main_test_loss]
        for factor in factors:
            train_loss, test_loss = load_losses(f"{surr_name.lower()}_sparse_{factor}")
            sparse_train_losses.append(train_loss)
            sparse_test_losses.append(test_loss)
        plot_losses(
            tuple(sparse_train_losses),
            tuple(f"Factor {factor}" for factor in [1] + factors),
            title="Sparse Losses",
            save=True,
            conf=conf,
            surr_name=surr_name,
            mode="sparse",
        )

    # UQ losses
    if conf["UQ"]["enabled"]:
        n_models = conf["UQ"]["n_models"]
        uq_train_losses = [main_train_loss]
        uq_test_losses = [main_test_loss]
        for i in range(1, n_models):
            train_loss, test_loss = load_losses(f"{surr_name.lower()}_UQ_{i}")
            uq_train_losses.append(train_loss)
            uq_test_losses.append(test_loss)
        plot_losses(
            tuple(uq_train_losses),
            tuple(f"Model {i}" for i in range(n_models)),
            title="UQ Losses",
            save=True,
            conf=conf,
            surr_name=surr_name,
            mode="UQ",
        )


def plot_losses(
    loss_histories: tuple[np.array, ...],
    labels: tuple[str, ...],
    title: str = "Losses",
    save: bool = False,
    conf: Optional[dict] = None,
    surr_name: Optional[str] = None,
    mode: str = "main",
) -> None:
    """
    Plot the loss trajectories for the training of multiple models.

    :param loss_histories: List of loss history arrays.
    :param labels: List of labels for each loss history.
    :param title: Title of the plot.
    :param save: Whether to save the plot as an image file.
    :param conf: The configuration dictionary.
    :param surr_name: The name of the surrogate model.
    :param mode: The mode of the training.
    """

    # Create the figure
    plt.figure(figsize=(12, 6))
    if len(loss_histories) != len(labels):
        plt.plot(loss_histories, label=labels)
    else:
        for loss, label in zip(loss_histories, labels):
            plt.plot(loss, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(title)
    plt.legend()

    if save and conf and surr_name:
        save_plot(plt, "losses_" + mode.lower() + ".png", conf, surr_name)

    # plt.show()
    plt.close()
