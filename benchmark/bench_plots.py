import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
import os


# Utility functions for plotting


def save_plot(plt, filename: str, conf: dict, surr_name: str = "") -> None:
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


# Per-surrogate model plots


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
    plt.ylim(1e-5, 1)
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
    metrics: np.ndarray,
    model_errors: np.ndarray,
    mode: str,
    save: bool = False,
) -> None:
    """
    Plot the generalization errors of a model for various metrics.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        metrics (np.ndarray): The metrics (e.g., intervals, cutoffs, batch sizes, number of training samples).
        model_errors (np.ndarray): The model errors.
        mode (str): The mode of generalization ("interpolation", "extrapolation", "sparse", "batchsize").
        save (bool): Whether to save the plot.

    Returns:
        None
    """
    if mode == "interpolation":
        xlabel = "Interpolation Interval"
        title = "Interpolation Errors"
        filename = "interpolation_errors.png"
    elif mode == "extrapolation":
        xlabel = "Extrapolation Cutoff"
        title = "Extrapolation Errors"
        filename = "extrapolation_errors.png"
    elif mode == "sparse":
        xlabel = "Number of Training Samples"
        title = "Sparse Training Errors"
        filename = "sparse_errors.png"
    elif mode == "batchsize":
        xlabel = "Batch Size"
        title = "Batch Size Training Errors"
        filename = "batchsize_errors.png"
    else:
        raise ValueError(
            "Invalid mode. Choose from 'interpolation', 'extrapolation', 'sparse', 'batchsize'."
        )

    plt.figure(figsize=(10, 6))
    plt.scatter(metrics, model_errors, label=surr_name)
    plt.xlabel(xlabel)
    if mode == "sparse" or mode == "batchsize":
        plt.xscale("log")
    plt.ylabel("Mean Squared Error")
    plt.yscale("log")
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.legend()

    if save and conf:
        save_plot(plt, filename, conf, surr_name)

    # plt.show()

    plt.close()


def plot_average_errors_over_time(
    surr_name: str,
    conf: dict,
    errors: np.ndarray,
    metrics: np.ndarray,
    timesteps: np.ndarray,
    mode: str,
    save: bool = False,
) -> None:
    """
    Plot the errors over time for different modes (interpolation, extrapolation, sparse, batchsize).

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        errors (np.ndarray): Errors array of shape [N_metrics, n_timesteps].
        metrics (np.ndarray): Metrics array of shape [N_metrics].
        timesteps (np.ndarray): Timesteps array.
        mode (str): The mode of evaluation ('interpolation', 'extrapolation', 'sparse', 'batchsize').
        save (bool, optional): Whether to save the plot as a file.
    """
    plt.figure(figsize=(10, 6))

    labels = {
        "interpolation": "interval",
        "extrapolation": "cutoff",
        "sparse": "samples",
        "batchsize": "batch size",
    }
    if mode not in labels:
        raise ValueError(
            "Invalid mode. Choose from 'interpolation', 'extrapolation', 'sparse', 'batchsize'."
        )

    colors = plt.cm.magma(np.linspace(0.15, 0.85, errors.shape[0]))

    if mode == "sparse":
        for i, metric in enumerate(metrics):
            label = f"{metric} {labels[mode]}"
            plt.plot(timesteps, errors[i], label=label, color=colors[i])
    else:
        for i, metric in enumerate(metrics):
            label = f"{labels[mode]} {metric}"
            plt.plot(timesteps, errors[i], label=label, color=colors[i])
            if mode == "extrapolation":
                cutoff_point = timesteps[metric - 1]
                plt.axvline(
                    x=cutoff_point, color=colors[i], linestyle="--", linewidth=0.8
                )
                plt.text(
                    cutoff_point,
                    errors[i, metric - 1],
                    f"{metric}",
                    color=colors[i],
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Absolute Error")
    title = f"{mode.capitalize()} Errors Over Time"
    filename = f"{mode}_errors_over_time.png"

    plt.title(title)
    plt.legend()

    if save and conf:
        save_plot(plt, filename, conf, surr_name)

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
    colors = plt.cm.viridis(np.linspace(0, 0.9, num_chemicals))

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

    plt.suptitle(
        r"Examplary Predictions with Uncertainty Intervals ($\mu \pm (1,2,3) \sigma$)"
    )
    plt.legend()
    plt.tight_layout()

    if save and conf:
        save_plot(plt, "UQ_predictions.png", conf, surr_name)

    plt.close()


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
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, preds_std, label="Average Uncertainty")
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


def plot_surr_losses(model, surr_name: str, conf: dict, timesteps: np.ndarray) -> None:
    """
    Plot the training and test losses for the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        timesteps (np.ndarray): The timesteps array.
    """
    training_id = conf["training_id"]

    def load_losses(model_identifier: str):
        model.load(training_id, surr_name, model_identifier=model_identifier)
        return model.train_loss, model.test_loss

    # Main model losses
    if conf["accuracy"]:
        main_train_loss, main_test_loss = load_losses(f"{surr_name.lower()}_main")
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
            tuple(f"Cutoff {cutoff}" for cutoff in cutoffs + [len(timesteps)]),
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

    # Batchsize losses
    if conf["batch_scaling"]["enabled"]:
        batch_sizes = conf["batch_scaling"]["sizes"]
        batch_train_losses = []
        batch_test_losses = []
        for batch_size in batch_sizes:
            train_loss, test_loss = load_losses(
                f"{surr_name.lower()}_batchsize_{batch_size}"
            )
            batch_train_losses.append(train_loss)
            batch_test_losses.append(test_loss)
        plot_losses(
            tuple(batch_train_losses),
            tuple(f"Batch Size {batch_size}" for batch_size in batch_sizes),
            title="Batch Size Losses",
            save=True,
            conf=conf,
            surr_name=surr_name,
            mode="batchsize",
        )


# def plot_error_distribution_per_chemical(
#     surr_name: str,
#     conf: dict,
#     errors: np.ndarray,
#     chemical_names: list = None,
#     num_chemicals: int = 10,
#     save: bool = True,
# ) -> None:
#     """
#     Plot the distribution of errors for each chemical as a smoothed histogram plot.

#     Args:
#         surr_name (str): The name of the surrogate model.
#         conf (dict): The configuration dictionary.
#         errors (np.ndarray): Errors array of shape [num_samples, num_timesteps, num_chemicals].
#         chemical_names (list, optional): List of chemical names for labeling the lines.
#         num_chemicals (int, optional): Number of chemicals to plot. Default is 10.
#         save (bool, optional): Whether to save the plot as a file.
#     """
#     # Reshape errors to combine samples and timesteps
#     total_chemicals = errors.shape[2]
#     errors = errors.reshape(-1, total_chemicals)
#     n_errors = len(errors.reshape(-1))

#     # Determine how many chemicals to plot
#     num_chemicals = min(num_chemicals, total_chemicals)
#     errors = errors[:, :num_chemicals]

#     # Initialize list to hold log-transformed non-zero errors and count zeros
#     log_errors = []
#     zero_counts = 0

#     # Transform error magnitudes to log-space and filter out zeros
#     for i in range(num_chemicals):
#         chemical_errors = errors[:, i]
#         non_zero_chemical_errors = chemical_errors[chemical_errors > 0]
#         log_errors.append(np.log10(non_zero_chemical_errors))
#         zero_counts += np.sum(chemical_errors == 0)

#     # Calculate the 5th and 95th percentiles in the log-space
#     min_percentiles = [np.percentile(err, 1) for err in log_errors if len(err) > 0]
#     max_percentiles = [np.percentile(err, 99) for err in log_errors if len(err) > 0]

#     global_min = np.min(min_percentiles)
#     global_max = np.max(max_percentiles)

#     # Set up the x-axis range to nearest whole numbers in log-space
#     x_min = np.floor(global_min)
#     x_max = np.ceil(global_max)

#     # Set up the plot
#     plt.figure(figsize=(10, 6))
#     colors = plt.cm.tab20b(np.linspace(0, 1, num_chemicals))

#     # Define the x-axis range for plotting
#     x_vals = np.linspace(x_min, x_max + 0.1, 100)

#     for i in range(num_chemicals):
#         # Compute histogram in log-space
#         hist, bin_edges = np.histogram(log_errors[i], bins=x_vals, density=True)

#         # Smooth the histogram with a Gaussian filter
#         smoothed_hist = gaussian_filter1d(hist, sigma=2)

#         # Normalize smoothed histogram so that the total area sums to 1
#         # smoothed_hist /= np.trapz(smoothed_hist, bin_edges[:-1])

#         # Plot the smoothed histogram
#         plt.plot(
#             10 ** bin_edges[:-1],
#             smoothed_hist,
#             label=(
#                 chemical_names[i]
#                 if chemical_names and len(chemical_names) > i
#                 else None
#             ),
#             color=colors[i],
#         )

#     plt.xscale("log")  # Log scale for error magnitudes
#     plt.xlim(10**x_min, 10**x_max)  # Set x-axis range based on log-space calculations
#     plt.xlabel("Magnitude of Error")
#     plt.ylabel("Density (PDF)")
#     plt.title(
#         f"Error Distribution per Chemical (Test Samples: {n_errors}, Excluded zeros: {zero_counts})"
#     )

#     if chemical_names:
#         plt.legend()

#     if save and conf:
#         save_plot(plt, "error_distribution_per_chemical.png", conf, surr_name)

#     plt.show()
#     plt.close()


def plot_error_distribution_per_chemical(
    surr_name: str,
    conf: dict,
    errors: np.ndarray,
    chemical_names: list = None,
    num_chemicals: int = 10,
    save: bool = True,
) -> None:
    """
    Plot the distribution of errors for each chemical as a smoothed histogram plot.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        errors (np.ndarray): Errors array of shape [num_samples, num_timesteps, num_chemicals].
        chemical_names (list, optional): List of chemical names for labeling the lines.
        num_chemicals (int, optional): Number of chemicals to plot. Default is 10.
        save (bool, optional): Whether to save the plot as a file.
    """
    # Reshape errors to combine samples and timesteps
    total_chemicals = errors.shape[2]
    errors = errors.reshape(-1, total_chemicals)
    n_errors = len(errors.reshape(-1))

    # Cap the number of chemicals to plot at 50
    num_chemicals = min(num_chemicals, 50)
    errors = errors[:, :num_chemicals]
    chemical_names = chemical_names[:num_chemicals] if chemical_names else None

    # Split the chemicals into groups of 10
    chemicals_per_plot = 10
    num_plots = int(np.ceil(num_chemicals / chemicals_per_plot))

    # Initialize list to hold log-transformed non-zero errors and count zeros
    log_errors = []
    zero_counts = 0

    # Transform error magnitudes to log-space and filter out zeros
    for i in range(num_chemicals):
        chemical_errors = errors[:, i]
        non_zero_chemical_errors = chemical_errors[chemical_errors > 0]
        log_errors.append(np.log10(non_zero_chemical_errors))
        zero_counts += np.sum(chemical_errors == 0)

    # Calculate the 1st and 99th percentiles in the log-space
    min_percentiles = [np.percentile(err, 1) for err in log_errors if len(err) > 0]
    max_percentiles = [np.percentile(err, 99) for err in log_errors if len(err) > 0]

    global_min = np.min(min_percentiles)
    global_max = np.max(max_percentiles)

    # Set up the x-axis range to nearest whole numbers in log-space
    x_min = np.floor(global_min)
    x_max = np.ceil(global_max)

    # Create subplots with shared x-axis
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even if there's only one plot

    colors = plt.cm.magma(np.linspace(0.15, 0.85, chemicals_per_plot))

    # Define the x-axis range for plotting
    x_vals = np.linspace(x_min, x_max + 0.1, 100)

    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        start_idx = plot_idx * chemicals_per_plot
        end_idx = min((plot_idx + 1) * chemicals_per_plot, num_chemicals)

        for i in range(start_idx, end_idx):
            # Compute histogram in log-space
            hist, bin_edges = np.histogram(log_errors[i], bins=x_vals, density=True)

            # Smooth the histogram with a Gaussian filter
            smoothed_hist = gaussian_filter1d(hist, sigma=2)

            # Plot the smoothed histogram
            ax.plot(
                10 ** bin_edges[:-1],
                smoothed_hist,
                label=(
                    chemical_names[i]
                    if chemical_names and len(chemical_names) > i
                    else None
                ),
                color=colors[i % chemicals_per_plot],
            )

        ax.set_yscale("linear")
        ax.set_ylabel("Density (PDF)")
        if chemical_names:
            ax.legend()

    plt.xscale("log")  # Log scale for error magnitudes
    plt.xlim(10**x_min, 10**x_max)  # Set x-axis range based on log-space calculations
    plt.xlabel("Magnitude of Error")
    fig.suptitle(
        f"Error Distribution per Chemical (Test Samples: {n_errors}, Excluded zeros: {zero_counts})"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save and conf:
        save_plot(plt, "error_distribution_per_chemical.png", conf, surr_name)

    plt.show()
    plt.close()


# Comparative plots from here on!


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

    # Colormap
    colors = plt.cm.magma(np.linspace(0.15, 0.85, len(loss_histories)))

    # Create the figure
    plt.figure(figsize=(12, 6))
    loss_plotted = False
    for loss, label in zip(loss_histories, labels):
        if loss is not None:
            plt.plot(loss, label=label, color=colors[labels.index(label)])
            loss_plotted = True
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(title)
    plt.legend()

    if not loss_plotted:
        plt.text(
            0.5,
            0.5,
            "No losses available",
            horizontalalignment="center",
            verticalalignment="center",
        )

    if save and conf and surr_name:
        save_plot(plt, "losses_" + mode.lower() + ".png", conf, surr_name)

    # plt.show()
    plt.close()


def plot_loss_comparison(
    train_losses: tuple[np.ndarray, ...],
    test_losses: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    config: dict,
    save: bool = True,
) -> None:
    """
    Plot the training and test losses for different surrogate models.

    Args:
        train_losses (tuple): Tuple of training loss arrays for each surrogate model.
        test_losses (tuple): Tuple of test loss arrays for each surrogate model.
        labels (tuple): Tuple of labels for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(train_losses)))

    for i, (train_loss, test_loss, label) in enumerate(
        zip(train_losses, test_losses, labels)
    ):
        plt.plot(train_loss, label=f"{label} Train Loss", color=colors[i])
        plt.plot(test_loss, label=f"{label} Test Loss", linestyle="--", color=colors[i])

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Comparison of Training and Test Losses")
    plt.legend()
    plt.grid(True)

    if save and config:
        save_plot(plt, "comparison_main_model_losses.png", config)

    plt.show()


def plot_MAE_comparison(
    MAEs: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    config: dict,
    save: bool = True,
) -> None:
    """
    Plot the MAE for different surrogate models.

    Args:
        MAE (tuple): Tuple of accuracy arrays for each surrogate model.
        labels (tuple): Tuple of labels for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.
    """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(MAEs)))

    for i, (accuracy, label) in enumerate(zip(MAEs, labels)):
        plt.plot(accuracy, label=label, color=colors[i])

    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.yscale("log")
    plt.title("Comparison of Model Mean Absolute Errors")
    plt.legend()
    plt.grid(True)

    if save and config:
        save_plot(plt, "comparison_main_model_MAE.png", config)

    plt.show()


def plot_MAE_comparison_train_duration(
    MAEs: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    train_durations: tuple[float, ...],
    config: dict,
    save: bool = True,
) -> None:
    """
    Plot the MAE for different surrogate models.

    Args:
        MAE (tuple): Tuple of accuracy arrays for each surrogate model.
        labels (tuple): Tuple of labels for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.
    """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(MAEs)))

    for i, (accuracy, label, train_duration) in enumerate(
        zip(MAEs, labels, train_durations)
    ):
        epoch_times = np.linspace(0, train_duration, len(accuracy))
        plt.plot(epoch_times, accuracy, label=label, color=colors[i])

    plt.xlabel("Time (s)")
    plt.ylabel("MAE")
    plt.yscale("log")
    plt.title("Comparison of Model Mean Absolute Error over Training Duration")
    plt.legend()
    plt.grid(True)

    if save and config:
        save_plot(plt, "comparison_main_model_MAE_time.png", config)

    plt.show()


def plot_relative_errors(
    mean_errors: dict[str, np.ndarray],
    median_errors: dict[str, np.ndarray],
    timesteps: np.ndarray,
    config: dict,
    save: bool = True,
) -> None:
    """
    Plot the relative errors over time for different surrogate models.

    Args:
        mean_errors (dict): dictionary containing the mean relative errors for each surrogate model.
        median_errors (dict): dictionary containing the median relative errors for each surrogate model.
        timesteps (np.ndarray): Array of timesteps.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(mean_errors)))
    linestyles = ["-", "--"]

    for i, surrogate in enumerate(mean_errors.keys()):
        plt.plot(
            timesteps,
            mean_errors[surrogate],
            label=f"{surrogate} Mean",
            color=colors[i],
            linestyle=linestyles[0],
        )
        plt.plot(
            timesteps,
            median_errors[surrogate],
            label=f"{surrogate} Median",
            color=colors[i],
            linestyle=linestyles[1],
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Relative Error")
    plt.yscale("log")
    plt.title("Comparison of Relative Errors Over Time")
    plt.legend()
    # plt.grid(True)

    if save and config:
        save_plot(plt, "comparison_relative_errors.png", config)

    plt.close()


def plot_uncertainty_over_time_comparison(
    uncertainties: dict[str, np.ndarray],
    timesteps: np.ndarray,
    config: dict,
    save: bool = True,
) -> None:
    """
    Plot the uncertainty over time for different surrogate models.

    Args:
        uncertainties (dict): Dictionary containing the uncertainties for each surrogate model.
        timesteps (np.ndarray): Array of timesteps.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.

    Returns:
        None
    """

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(uncertainties)))
    for i, surrogate in enumerate(uncertainties.keys()):
        plt.plot(timesteps, uncertainties[surrogate], label=surrogate, color=colors[i])

    plt.xlabel("Timesteps")
    plt.ylabel("Uncertainty")
    plt.title("Comparison of Predictive Uncertainty Over Time")
    plt.legend()

    if save and config:
        save_plot(plt, "comparison_uncertainty_over_time.png", config)

    plt.close()


def inference_time_bar_plot(
    surrogates: list[str],
    means: list[float],
    stds: list[float],
    config: dict,
    save: bool = True,
) -> None:
    """
    Plot the mean inference time with standard deviation for different surrogate models.

    Args:
        surrogates (List[str]): List of surrogate model names.
        means (List[float]): List of mean inference times for each surrogate model.
        stds (List[float]): List of standard deviation of inference times for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool, optional): Whether to save the plot. Defaults to True.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(surrogates)))
    plt.bar(
        surrogates, means, yerr=stds, capsize=5, alpha=0.7, color=colors, ecolor="black"
    )
    plt.xlabel("Surrogate Model")
    plt.ylabel("Mean Inference Time per Prediction (s)")
    plt.yscale("log")
    plt.title("Comparison of Mean Inference Time with Standard Deviation")

    if save:
        save_plot(plt, "comparison_mean_inference_time.png", config)

    plt.close()


def plot_generalization_error_comparison(
    surrogates: list[str],
    metrics_list: list[np.array],
    model_errors_list: list[np.array],
    xlabel: str,
    filename: str,
    config: dict,
    save: bool = True,
    xlog: bool = False,
) -> None:
    """
    Plot the generalization errors of different surrogate models.

    Args:
        surrogates (list): List of surrogate model names.
        metrics_list (list[np.array]): List of numpy arrays containing the metrics for each surrogate model.
        model_errors_list (list[np.array]): List of numpy arrays containing the errors for each surrogate model.
        xlabel (str): Label for the x-axis.
        filename (str): Filename to save the plot.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.
        xlog (bool): Whether to use a log scale for the x-axis.

    Returns:
        None
    """
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(surrogates)))

    plt.figure(figsize=(10, 6))
    for i, surrogate in enumerate(surrogates):
        plt.scatter(
            metrics_list[i], model_errors_list[i], label=surrogate, color=colors[i]
        )

    plt.xlabel(xlabel)
    if xlog:
        plt.xscale("log")
    plt.ylabel("Mean Absolute Error")
    plt.yscale("log")
    plt.title(f"Comparison of {xlabel} Errors")
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.legend()

    if save:
        save_plot(plt, filename, config)

    plt.show()
    plt.close()


def plot_error_correlation_heatmap(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    errors: np.ndarray,
    save: bool = False,
    threshold_factor: float = 1e-4,
) -> None:
    """
    Plot the correlation between predictive uncertainty and prediction errors using a heatmap.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        preds_std (np.ndarray): Standard deviation of predictions from the ensemble of models.
        errors (np.ndarray): Prediction errors.
        save (bool, optional): Whether to save the plot as a file.
        threshold_factor (float, optional): Fraction of max value below which cells are set to 0. Default is 0.001.
    """

    plt.figure(figsize=(10, 6))
    heatmap, xedges, yedges = np.histogram2d(
        preds_std.flatten(), errors.flatten(), bins=50
    )

    # Apply threshold to create a mask and determine new axis limits
    max_value = heatmap.max()
    threshold = threshold_factor * max_value

    # Mask to find the densest part for axis limits, but don't alter heatmap values
    mask = heatmap >= threshold
    non_zero_indices = np.nonzero(mask)
    x_min, x_max = (
        xedges[non_zero_indices[0].min()],
        xedges[non_zero_indices[0].max() + 1],
    )
    y_min, y_max = (
        yedges[non_zero_indices[1].min()],
        yedges[non_zero_indices[1].max() + 1],
    )

    # Find the common range to use for both axes to ensure aspect ratio
    axis_min = min(x_min, y_min)
    axis_max = max(x_max, y_max)

    # Re-bin the data using the new axis limits
    heatmap, xedges, yedges = np.histogram2d(
        preds_std.flatten(),
        errors.flatten(),
        bins=50,
        range=[[axis_min, axis_max], [axis_min, axis_max]],
    )

    # Set all fields with 0 counts to 1 count for log scale plotting
    heatmap[heatmap == 0] = 1

    plt.imshow(
        np.log10(heatmap.T),  # Log scale for better visualization
        origin="lower",
        aspect="auto",
        extent=[axis_min, axis_max, axis_min, axis_max],
        cmap="inferno",
    )
    plt.colorbar(label=r"$\log_{10}$(Counts)")
    plt.xlabel("Predictive Uncertainty")
    plt.ylabel("Prediction Error")
    plt.title(
        "Correlation between Predictive Uncertainty and Prediction Errors (Heatmap)"
    )

    # Add diagonal line
    plt.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color="white",
        linestyle="--",
        linewidth=1,
    )

    if save and conf:
        save_plot(plt, "uncertainty_vs_errors_heatmap.png", conf, surr_name)

    plt.show()
    plt.close()


def plot_dynamic_correlation_heatmap(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    errors: np.ndarray,
    save: bool = False,
    threshold_factor: float = 1e-4,
    xcut_percent: float = 1e-3,  # Default to keep 95% of the total counts in the heatmap
) -> None:
    """
    Plot the correlation between predictive uncertainty and prediction errors using a heatmap.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        preds_std (np.ndarray): Standard deviation of predictions from the ensemble of models.
        errors (np.ndarray): Prediction errors.
        save (bool, optional): Whether to save the plot as a file.
        threshold_factor (float, optional): Fraction of max value below which cells are set to 0. Default is 5e-5.
        cutoff_percent (float, optional): The percentage of total counts to include in the heatmap. Default is 0.95.
    """

    plt.figure(figsize=(10, 6))
    heatmap, xedges, yedges = np.histogram2d(
        preds_std.flatten(), errors.flatten(), bins=50
    )

    # Apply threshold to create a mask and determine new axis limits
    max_value = heatmap.max()
    threshold = threshold_factor * max_value

    # Mask to find the densest part for y-axis limits
    mask = heatmap >= threshold
    non_zero_indices = np.nonzero(mask)
    y_min, y_max = (
        yedges[non_zero_indices[1].min()],
        yedges[non_zero_indices[1].max() + 1],
    )

    # Calculate the cumulative counts along the x-axis
    cumulative_counts = np.cumsum(heatmap.sum(axis=1))
    total_counts = cumulative_counts[-1]
    cutoff_percent = 1 - xcut_percent
    cutoff_count = cutoff_percent * total_counts

    # Find the x_max where the cumulative count exceeds cutoff_count
    x_max_index = np.searchsorted(cumulative_counts, cutoff_count)
    x_max = xedges[x_max_index + 1]  # +1 to include the bin

    x_min = 0  # Normalized gradients typically start from 0

    # Re-bin the data using the new axis limits
    heatmap, xedges, yedges = np.histogram2d(
        preds_std.flatten(),
        errors.flatten(),
        bins=50,
        range=[[x_min, x_max], [y_min, y_max]],
    )

    # Set all fields with 0 counts to 1 count for log scale plotting
    heatmap[heatmap == 0] = 1

    plt.imshow(
        np.log10(heatmap.T),  # Log scale for better visualization
        origin="lower",
        aspect="auto",
        extent=[x_min, x_max, y_min, y_max],
        cmap="inferno",
    )
    plt.colorbar(label=r"$\log_{10}$(Counts)")
    plt.xlabel("Absolute Gradient (Normalized)")
    plt.ylabel("Absolute Prediction Error")
    plt.title("Correlation between Gradients and Prediction Errors")

    if save and conf:
        save_plot(plt, "dynamic_correlation_heatmap.png", conf, surr_name)

    plt.show()
    plt.close()


def plot_error_distribution_comparative(
    errors: dict[str, np.ndarray],
    conf: dict,
    save: bool = True,
) -> None:
    """
    Plot the comparative distribution of errors for each surrogate model as a smoothed histogram plot.

    Args:
        conf (dict): The configuration dictionary.
        errors (dict): Dictionary containing numpy arrays of shape [num_samples, num_timesteps, num_chemicals] for each model.
        save (bool, optional): Whether to save the plot as a file.
    """
    # Number of models
    model_names = list(errors.keys())
    num_models = len(model_names)

    # Initialize list to hold log-transformed non-zero errors and count zeros
    log_errors = []
    zero_counts = 0

    # Transform error magnitudes to log-space and filter out zeros for each model
    for model_name in model_names:
        model_errors = errors[model_name].flatten()
        non_zero_model_errors = model_errors[model_errors > 0]
        log_errors.append(np.log10(non_zero_model_errors))
        zero_counts += np.sum(model_errors == 0)

    # Calculate the 1st and 99th percentiles in the log-space
    min_percentiles = [np.percentile(err, 3) for err in log_errors if len(err) > 0]
    max_percentiles = [np.percentile(err, 97) for err in log_errors if len(err) > 0]

    global_min = np.min(min_percentiles)
    global_max = np.max(max_percentiles)

    # Set up the x-axis range to nearest whole numbers in log-space
    x_min = np.floor(global_min)
    x_max = np.ceil(global_max)

    # Set up the plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_models))

    # Define the x-axis range for plotting
    x_vals = np.linspace(x_min, x_max + 0.1, 100)

    for i, model_name in enumerate(model_names):
        # Compute histogram in log-space
        hist, bin_edges = np.histogram(log_errors[i], bins=x_vals, density=True)

        # Smooth the histogram with a Gaussian filter
        smoothed_hist = gaussian_filter1d(hist, sigma=2)

        # Plot the smoothed histogram
        plt.plot(
            10 ** bin_edges[:-1],
            smoothed_hist,
            label=model_name,
            color=colors[i],
        )

    plt.xscale("log")  # Log scale for error magnitudes
    plt.xlim(10**x_min, 10**x_max)  # Set x-axis range based on log-space calculations
    plt.xlabel("Magnitude of Error")
    plt.ylabel("Density (PDF)")
    plt.title(
        f"Error Distribution per Model (Test Samples: {len(errors[model_names[0]].flatten())}, Excluded zeros: {zero_counts})"
    )

    plt.legend()

    if save and conf:
        save_plot(plt, "error_distribution_comparative.png", conf)

    plt.show()
    plt.close()


def rbf_kernel(x, y, bandwidth):
    """
    Compute the RBF (Gaussian) kernel between two arrays.

    Args:
        x (np.ndarray): First array.
        y (np.ndarray): Second array.
        bandwidth (float): Bandwidth parameter for the RBF kernel.

    Returns:
        np.ndarray: The RBF kernel matrix.
    """
    sq_dists = cdist(x, y, "sqeuclidean")
    return np.exp(-sq_dists / (2 * bandwidth**2))


def plot_correlation_KDE(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    errors: np.ndarray,
    save: bool = False,
    bandwidth: float = 0.01,
) -> None:
    """
    Plot the correlation between predictive uncertainty and prediction errors using a KDE plot.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        preds_std (np.ndarray): Standard deviation of predictions from the ensemble of models.
        errors (np.ndarray): Prediction errors.
        save (bool, optional): Whether to save the plot as a file.
        bandwidth (float, optional): Bandwidth for the RBF kernel.
    """
    # Normalize the errors
    errors = errors / np.abs(errors).max()

    # Prepare data
    data = np.vstack([preds_std.flatten(), errors.flatten()]).T

    # Create grid for evaluation
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    x_grid, y_grid = np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T

    # Compute KDE
    kde_values = (
        rbf_kernel(grid_points, data, bandwidth).mean(axis=1).reshape(x_mesh.shape)
    )

    # Plot KDE
    plt.figure(figsize=(10, 6))
    plt.imshow(
        np.log10(kde_values.T),
        origin="lower",
        aspect="auto",
        extent=[x_min, x_max, y_min, y_max],
        cmap="viridis",
    )
    plt.colorbar(label=r"$\log_{10}$(Density)")
    plt.xlabel("Predictive Uncertainty")
    plt.ylabel("Prediction Error (Normalized)")
    plt.title("Correlation between Predictive Uncertainty and Prediction Errors (KDE)")

    if save and conf:
        save_plot(plt, "correlation_KDE.png", conf, surr_name)

    plt.close()
