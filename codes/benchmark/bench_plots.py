import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

from .bench_utils import format_time

# Utility functions for plotting


def save_plot(
    plt,
    filename: str,
    conf: dict,
    surr_name: str = "",
    dpi: int = 300,
    base_dir: str = "plots",  # Base directory for saving plots
    increase_count: bool = False,  # Whether to increase the count for existing filenames
) -> None:
    """
    Save the plot to a file, creating necessary directories if they don't exist.

    Args:
        plt (matplotlib.pyplot): The plot object to save.
        filename (str): The desired filename for the plot.
        conf (dict): The configuration dictionary.
        surr_name (str): The name of the surrogate model.
        dpi (int): The resolution of the saved plot.
        base_dir (str, optional): The base directory where plots will be saved. Default is "plots".
        increase_count (bool, optional): Whether to increment the filename count if a file already exists. Default is True.

    Raises:
        ValueError: If the configuration dictionary does not contain the required keys.
    """
    if "training_id" not in conf:
        raise ValueError("Configuration dictionary must contain 'training_id'.")

    training_id = conf["training_id"]
    plot_dir = os.path.join(base_dir, training_id, surr_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Call save_plot_counter with increase_count option
    filepath = save_plot_counter(filename, plot_dir, increase_count=increase_count)
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    if conf["verbose"]:
        print(f"Plot saved as: {filepath}")


def save_plot_counter(
    filename: str, directory: str, increase_count: bool = True
) -> str:
    """
    Save a plot with an incremented filename if a file with the same name already exists.

    Args:
        filename (str): The desired filename for the plot.
        directory (str): The directory to save the plot in.
        increase_count (bool, optional): Whether to increment the filename count if a file already exists. Default is True.

    Returns:
        str: The full path to the saved plot.
    """
    if not increase_count:
        filepath = os.path.join(directory, filename)
        return filepath

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
    mean = np.mean(mean_errors)
    median_errors = np.median(relative_errors, axis=(0, 2))
    median = np.median(median_errors)
    p50_upper = np.percentile(relative_errors, 75, axis=(0, 2))
    p50_lower = np.percentile(relative_errors, 25, axis=(0, 2))
    p90_upper = np.percentile(relative_errors, 95, axis=(0, 2))
    p90_lower = np.percentile(relative_errors, 5, axis=(0, 2))
    p99_upper = np.percentile(relative_errors, 99.5, axis=(0, 2))
    p99_lower = np.percentile(relative_errors, 0.5, axis=(0, 2))

    timesteps = np.arange(relative_errors.shape[1])

    plt.figure(figsize=(10, 6))
    mean_label = f"Mean Error (Mean: {mean*100:.2f} %)"
    plt.plot(timesteps, mean_errors, label=mean_label, color="blue")
    median_label = f"Median Error (Median: {median*100:.2f} %)"
    plt.plot(timesteps, median_errors, label=median_label, color="red")

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
    plt.ylabel("Relative Error")
    plt.xlim(timesteps[0], timesteps[-1])
    plt.title(title)
    plt.legend()

    if save and conf:
        save_plot(plt, "accuracy_rel_errors_time.png", conf, surr_name)

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
        save_plot(plt, "gradient_error_correlation.png", conf, surr_name)

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
    plt.scatter(metrics, model_errors, label=surr_name, color="#3A1A5A")
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

    plt.xlabel("Time")
    plt.xlim(timesteps[0], timesteps[-1])
    plt.ylabel("Mean Absolute Error")
    title = f"Mean Absolute Errors over Time ({mode.capitalize()}, {surr_name})"
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
    example_idx: int = 0,
    num_chemicals: int = 100,
    labels: list[str] | None = None,
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
        example_idx (int, optional): Index of the example to plot. Default is 0.
        num_chemicals (int, optional): Number of chemicals to plot. Default is 100.
        labels (list, optional): List of labels for the chemicals.
        save (bool, optional): Whether to save the plot as a file.
    """
    # Cap the number of chemicals at 100
    num_chemicals = min(preds_std.shape[2], num_chemicals)

    # Determine the number of plots needed
    chemicals_per_plot = 10
    num_plots = int(np.ceil(num_chemicals / chemicals_per_plot))

    # Define the color palette
    colors = plt.cm.viridis(np.linspace(0, 1, chemicals_per_plot))

    # Create subplots
    fig = plt.figure(figsize=(12, 6 * num_plots))
    gs = GridSpec(num_plots, 1, figure=fig)  # Single column of plots

    for plot_idx in range(num_plots):
        ax = fig.add_subplot(gs[plot_idx])

        start_idx = plot_idx * chemicals_per_plot
        end_idx = min((plot_idx + 1) * chemicals_per_plot, num_chemicals)

        legend_lines = []  # To store the line objects for the legend

        for chem_idx in range(start_idx, end_idx):
            color = colors[chem_idx % chemicals_per_plot]
            gt = targets[example_idx, :, chem_idx]
            mean = preds_mean[example_idx, :, chem_idx]
            std = preds_std[example_idx, :, chem_idx]

            # Plot ground truth and store the line object
            (gt_line,) = ax.plot(timesteps, gt, "--", color=color)
            # Plot prediction mean and store the line object
            ax.plot(timesteps, mean, "-", color=color)

            # Store only the ground truth line in legend_lines
            legend_lines.append(gt_line)

            # Plot standard deviations as shaded areas
            for sigma_multiplier in [1, 2, 3]:  # 1, 2, and 3 standard deviations
                ax.fill_between(
                    timesteps,
                    mean - sigma_multiplier * std,
                    mean + sigma_multiplier * std,
                    color=color,
                    alpha=0.5 / sigma_multiplier,
                )

        # Set the y-axis label for each subplot
        ax.set_ylabel("log(Chemical Abundance)")

        # Create a legend directly next to the plot using the stored line objects
        if labels is not None:
            legend_labels = labels[start_idx:end_idx]
            ax.legend(
                legend_lines,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),  # Position legend to the right of the plot
                bbox_transform=ax.transAxes,  # Place it relative to the axes
                title="Chemical Labels",
            )

        # Set the x limit exactly from the lowest to the highest timestep
        ax.set_xlim(timesteps.min(), timesteps.max())

    # Add a single x-axis label to the bottom plot
    fig.text(0.5, 0.04, "Time", ha="center", va="center", fontsize=12)

    # Create a general legend for line styles, positioned below the title
    handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", label="Ground Truth"),
        plt.Line2D(
            [0], [0], color="black", linestyle="-", label="Prediction (Ensemble Mean)"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),  # Adjust to position the legend below the title
        ncol=2,
        fontsize="small",
    )

    # Adjust the title to include the sample index and place additional information on a second line
    plt.suptitle(
        f"DeepEnsemble: Example Predictions with Uncertainty for {surr_name} \n"
        f"Sample Index: {example_idx},  " + r"$\mu \pm (1,2,3) \sigma$ Intervals",
        y=0.97,
    )

    plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.92])

    if save and conf:
        save_plot(plt, "uncertainty_deepensemble_preds.png", conf, surr_name, dpi=300)

    plt.close()


def plot_average_uncertainty_over_time(
    surr_name: str,
    conf: dict,
    errors_time: np.ndarray,
    preds_std: np.ndarray,
    timesteps: np.ndarray,
    save: bool = False,
) -> None:
    """
    Plot the average uncertainty over time.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        errors_time (np.ndarray): Prediction errors over time.
        preds_std (np.ndarray): Standard deviation over time of predictions from the ensemble of models.
        timesteps (np.ndarray): Timesteps array.
        save (bool, optional): Whether to save the plot as a file.
    """
    errors_mean = np.mean(errors_time)
    preds_mean = np.mean(preds_std)
    plt.figure(figsize=(10, 6))
    errors_label = f"Mean Uncertainty (Mean: {preds_mean:.2e})"
    preds_label = f"Mean Absolute Error (Mean: {errors_mean:.2e})"
    plt.plot(timesteps, preds_std, label=preds_label, color="#3A1A5A")
    plt.plot(timesteps, errors_time, label=errors_label, color="#DA5F4D")
    plt.xlabel("Time")
    plt.ylabel("Average Uncertainty / Mean Absolute Error")
    plt.xlim(timesteps[0], timesteps[-1])
    plt.title("Average Uncertainty and Mean Absolute Error Over Time")
    plt.legend()

    if save and conf:
        save_plot(plt, "uncertainty_over_time.png", conf, surr_name)

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
    if conf["uncertainty"]["enabled"]:
        n_models = conf["uncertainty"]["ensemble_size"]
        uq_train_losses = [main_train_loss]
        uq_test_losses = [main_test_loss]
        for i in range(n_models - 1):
            train_loss, test_loss = load_losses(f"{surr_name.lower()}_UQ_{i+1}")
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


def plot_error_distribution_per_chemical(
    surr_name: str,
    conf: dict,
    errors: np.ndarray,
    chemical_names: list[str] | None = None,
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
    # n_errors = len(errors.reshape(-1))

    # Cap the number of chemicals to plot at 50
    num_chemicals = min(num_chemicals, 50)
    errors = errors[:, :num_chemicals]
    chemical_names = (
        chemical_names[:num_chemicals] if chemical_names is not None else None
    )

    # Split the chemicals into groups of 10
    chemicals_per_plot = 10
    num_plots = int(np.ceil(num_chemicals / chemicals_per_plot))

    # Initialize list to hold log-transformed non-zero errors and count zeros
    log_errors = []
    zero_counts = 0

    # Transform error magnitudes to log-space and filter out zeros
    for i in range(num_chemicals):
        chemical_errors = errors[:, i]
        if np.isnan(chemical_errors).any():
            raise ValueError("Error values contain NaNs.")
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

    # colors = plt.cm.magma(np.linspace(0, 0.9, chemicals_per_plot))
    colors = get_custom_palette(chemicals_per_plot)

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
                    if chemical_names is not None and len(chemical_names) > i
                    else None
                ),
                color=colors[i % chemicals_per_plot],
            )

        ax.set_yscale("linear")
        ax.set_ylabel("Density (PDF)")
        if chemical_names is not None:
            ax.legend()

    plt.xscale("log")  # Log scale for error magnitudes
    plt.xlim(10**x_min, 10**x_max)  # Set x-axis range based on log-space calculations
    plt.xlabel("Relative Error")
    # Temp!
    fig.suptitle(f"Relative Error Distribution per Chemical ({surr_name})")
    # fig.suptitle(
    #     f"Error Distribution per Chemical (Test Samples: {n_errors}, Excluded zeros: {zero_counts})"
    # )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save and conf:
        save_plot(plt, "accuracy_error_per_quantity.png", conf, surr_name)

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
    plt.xlim(0, len(loss_histories[0]))
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
        # plt.plot(train_loss, label=f"{label} Train Loss", color=colors[i])
        plt.plot(test_loss, label=f"{label} Test Loss", linestyle="--", color=colors[i])

    max_epochs = max(len(loss) for loss in train_losses)

    plt.xlabel("Epoch")
    plt.xlim(0, max_epochs)
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Comparison of Training and Test Losses")
    plt.legend()
    plt.grid(True)

    if save and config:
        save_plot(plt, "losses_main_model.png", config)

    plt.close()


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
        # Modify the label to include the final MAE
        final_MAE = accuracy[-1]
        label = f"{label} (final MAE: {final_MAE:.2e})"
        plt.plot(accuracy, label=label, color=colors[i])

    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.yscale("log")
    plt.title("Comparison of Model Mean Absolute Errors")
    plt.legend()
    plt.grid(True)

    if save and config:
        save_plot(plt, "MAE_main_model.png", config)

    plt.close()


def plot_MAE_comparison_train_duration(
    MAEs: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    train_durations: tuple[float, ...],
    config: dict,
    save: bool = True,
) -> None:
    """
    Plot the MAE for different surrogate models over the course of training.

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
    plt.xlim(0, max(train_durations))
    plt.ylabel("MAE")
    plt.yscale("log")
    plt.title("Comparison of Model Mean Absolute Error over Training Duration")
    plt.legend()
    plt.grid(True)

    if save and config:
        save_plot(plt, "losses_MAE_main_model.png", config)

    plt.close()


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
        mean = np.mean(mean_errors[surrogate])
        mean_label = f"{surrogate} Mean = {mean*100:.2f} %"
        plt.plot(
            timesteps,
            mean_errors[surrogate],
            label=mean_label,
            color=colors[i],
            linestyle=linestyles[0],
        )
        median = np.mean(median_errors[surrogate])
        median_label = f"{surrogate} Median = {median*100:.2f} %"
        plt.plot(
            timesteps,
            median_errors[surrogate],
            label=median_label,
            color=colors[i],
            linestyle=linestyles[1],
        )

    plt.xlabel("Time")
    plt.xlim(timesteps[0], timesteps[-1])
    plt.ylabel("Relative Error")
    plt.yscale("log")
    plt.title("Comparison of Relative Errors Over Time")
    plt.legend()
    # plt.grid(True)

    if save and config:
        save_plot(plt, "accuracy_rel_errors_time_models.png", config)

    plt.close()


def plot_uncertainty_over_time_comparison(
    uncertainties: dict[str, np.ndarray],
    absolute_errors: dict[str, np.ndarray],
    timesteps: np.ndarray,
    config: dict,
    save: bool = True,
) -> None:
    """
    Plot the uncertainty over time for different surrogate models.

    Args:
        uncertainties (dict): Dictionary containing the uncertainties for each surrogate model.
        absolute_errors (dict): Dictionary containing the absolute errors for each surrogate model.
        timesteps (np.ndarray): Array of timesteps.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.

    Returns:
        None
    """

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(uncertainties)))
    for i, surrogate in enumerate(uncertainties.keys()):
        avg_uncertainty = np.mean(uncertainties[surrogate])
        uq_label = f"{surrogate} uncertainty (mean: {avg_uncertainty:.2e})"
        plt.plot(
            timesteps,
            uncertainties[surrogate],
            label=uq_label,
            linestyle="-",
            color=colors[i],
        )

        abs_errors = absolute_errors[surrogate]
        abs_errors_time = np.mean(abs_errors, axis=(0, 2))
        abs_error_avg = np.mean(abs_errors_time)
        err_label = f"{surrogate} abs. error (mean: {abs_error_avg:.2e})"
        plt.plot(
            timesteps,
            abs_errors_time,
            label=err_label,
            color=colors[i],
            linestyle="--",
        )

    plt.xlabel("Time")
    plt.xlim(timesteps[0], timesteps[-1])
    plt.ylabel("Uncertainty/Absolute Error")
    plt.yscale("log")
    plt.title("Comparison of Predictive Uncertainty Over Time")
    plt.legend()

    if save and config:
        save_plot(plt, "uncertainty_over_time.png", config)

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
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(surrogates)))
    ax.bar(
        surrogates, means, yerr=stds, capsize=5, alpha=0.7, color=colors, ecolor="black"
    )

    # Calculate the upper y-limit to provide space for text
    max_bar = max(means[i] + stds[i] for i in range(len(means)))
    # min_bar = min(means[i] - stds[i] for i in range(len(means)))
    # Temp!
    # ax.set_ylim(min_bar * 0.3, max_bar * 2)  # Set limits with some padding
    ax.set_ylim(0, max_bar * 1.2)  # Set limits with some padding

    # Add inference time as text to the bars using the format_time function
    for i, (mean, std) in enumerate(zip(means, stds)):
        formatted_time = format_time(mean, std)
        ax.text(
            i,
            mean + std * 1.2,  # Position the text above the bar
            formatted_time,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Surrogate Model")
    ax.set_ylabel("Mean Inference Time per Run")
    # Temp!
    # ax.set_yscale("log")
    ax.set_title("Surrogate Mean Inference Time Comparison")

    if save:
        save_plot(plt, "timing_inference.png", config)

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
        # Add thin lines of the same color to show the trend

        plt.plot(
            metrics_list[i],
            model_errors_list[i],
            color=colors[i],
            alpha=0.5,
            linestyle="-",
        )

    plt.xlabel(xlabel)
    if xlog:
        plt.xscale("log")
    plt.ylabel("Mean Absolute Error")
    plt.yscale("log")
    plt.title(f"Comparison of {xlabel} Errors")
    plt.grid(True, which="major", linestyle="--", linewidth=0.5, axis="y")
    plt.legend()

    if save:
        save_plot(plt, filename, config)

    plt.close()


def plot_error_correlation_heatmap(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    errors: np.ndarray,
    average_correlation: float,
    save: bool = False,
    threshold_factor: float = 1e-2,
) -> None:
    """
    Plot the correlation between predictive uncertainty and prediction errors using a heatmap.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        preds_std (np.ndarray): Standard deviation of predictions from the ensemble of models.
        errors (np.ndarray): Prediction errors.
        average_correlation (float): The average correlation between gradients and prediction errors (pearson correlation).
        save (bool, optional): Whether to save the plot as a file.
        threshold_factor (float, optional): Fraction of max value below which cells are set to 0. Default is 0.001.
    """

    plt.figure(figsize=(10, 6))
    heatmap, xedges, yedges = np.histogram2d(
        preds_std.flatten(), errors.flatten(), bins=100
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
        bins=100,
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
    title = f"Correlation between Predictive Uncertainty and Prediction Errors (Heatmap)\nAverage Correlation: {average_correlation:.2f}"
    plt.title(title)

    # Add diagonal line
    plt.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color="white",
        linestyle="--",
        linewidth=1,
    )

    if save and conf:
        save_plot(plt, "uncertainty_errors_correlation.png", conf, surr_name)

    plt.close()

    return max_value, axis_max


def plot_dynamic_correlation_heatmap(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    errors: np.ndarray,
    average_correlation: float,
    save: bool = False,
    threshold_factor: float = 1e-4,
    xcut_percent: float = 3e-3,  # Default to keep 95% of the total counts in the heatmap
) -> None:
    """
    Plot the correlation between predictive uncertainty and prediction errors using a heatmap.

    Args:
        surr_name (str): The name of the surrogate model.
        conf (dict): The configuration dictionary.
        preds_std (np.ndarray): Standard deviation of predictions from the ensemble of models.
        errors (np.ndarray): Prediction errors.
        average_correlation (float): The average correlation between gradients and prediction errors (pearson correlation).
        save (bool, optional): Whether to save the plot as a file.
        threshold_factor (float, optional): Fraction of max value below which cells are set to 0. Default is 5e-5.
        cutoff_percent (float, optional): The percentage of total counts to include in the heatmap. Default is 0.95.
    """

    plt.figure(figsize=(10, 6))
    heatmap, xedges, yedges = np.histogram2d(
        preds_std.flatten(), errors.flatten(), bins=100
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
        bins=100,
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
    title = f"Correlation between Gradients and Prediction Errors (Heatmap)\nAverage Correlation: {average_correlation:.2f}"
    plt.title(title)

    if save and conf:
        save_plot(plt, "gradient_error_heatmap.png", conf, surr_name)

    plt.close()

    return max_value, x_max, y_max


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
    mean_errors = []
    median_errors = []
    zero_counts = 0

    # Transform error magnitudes to log-space and filter out zeros for each model
    for model_name in model_names:
        model_errors = errors[model_name].flatten()
        mean_errors.append(np.mean(model_errors))
        median_errors.append(np.median(model_errors))
        non_zero_model_errors = model_errors[model_errors > 0]
        log_errors.append(np.log10(non_zero_model_errors))
        zero_counts += np.sum(model_errors == 0)

    # Calculate the 1st and 99th percentiles in the log-space
    min_percentiles = [np.percentile(err, 2) for err in log_errors if len(err) > 0]
    max_percentiles = [np.percentile(err, 98) for err in log_errors if len(err) > 0]

    global_min = np.min(min_percentiles)
    global_max = np.max(max_percentiles)

    # Set up the x-axis range to nearest whole numbers in log-space
    x_min = np.floor(global_min)
    x_max = np.ceil(global_max)

    # Set up the plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, num_models))

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

        # Plot the average error as a vertical line
        plt.axvline(
            x=mean_errors[i],
            color=colors[i],
            linestyle="--",
            label=f"{model_name} mean = {mean_errors[i]:.2e}",
        )

        plt.axvline(
            x=median_errors[i],
            color=colors[i],
            linestyle="-.",
            label=f"{model_name} median = {median_errors[i]:.2e}",
        )

    plt.xscale("log")  # Log scale for error magnitudes
    plt.xlim(10**x_min, 10**x_max)  # Set x-axis range based on log-space calculations
    plt.xlabel("Magnitude of Error")
    plt.ylabel("Density (PDF)")
    # num_test_samples = len(errors[model_names[0]].flatten()) * num_models
    # Temp!
    # plt.title(
    #     f"Error Distribution per Model (Test Samples: {num_test_samples}, Excluded zeros: {zero_counts})"
    # )
    plt.title("Comparison of Surrogate Relative Error Distributions")

    plt.legend()

    if save and conf:
        save_plot(plt, "accuracy_error_distributions.png", conf)

    plt.close()


def plot_comparative_error_correlation_heatmaps(
    preds_std: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    avg_correlations: dict[str, float],
    axis_max: dict[str, float],
    max_count: dict[str, float],
    config: dict,
    save: bool = True,
) -> None:
    """
    Plot comparative heatmaps of correlation between predictive uncertainty and prediction errors
    for multiple surrogate models.

    Args:
        preds_std (dict[str, np.ndarray]): Dictionary of standard deviation of predictions from the ensemble of models.
        errors (dict[str, np.ndarray]): Dictionary of prediction errors.
        avg_correlations (dict[str, float]): Dictionary of average correlations between gradients and prediction errors.
        axis_max (dict[str, float]): Dictionary of maximum values for axis scaling across models.
        max_count (dict[str, float]): Dictionary of maximum count values for heatmap normalization across models.
        config (dict): Configuration dictionary.
        save (bool, optional): Whether to save the plot. Defaults to True.

    Returns:
        None
    """
    # Number of models
    model_names = list(preds_std.keys())
    num_models = len(model_names)

    # Determine the global maximum value for axis scaling
    global_axis_max = max(axis_max.values())

    # Determine global max for heatmap normalization
    global_max_count = max(max_count.values())

    # Create subplots, one row per model
    fig, axes = plt.subplots(
        num_models, 1, figsize=(8, 6 * num_models), sharex=True, sharey=True
    )

    # Adjust layout for shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position for the colorbar

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        preds_std_model = preds_std[model_name].flatten()
        errors_model = errors[model_name].flatten()
        avg_corr = avg_correlations[model_name]

        # Compute 2D histogram (heatmap)
        heatmap, _, _ = np.histogram2d(
            preds_std_model,
            errors_model,
            bins=100,
            range=[[0, global_axis_max], [0, global_axis_max]],
        )

        # Set all fields with 0 counts to 1 count for log scale plotting
        heatmap[heatmap == 0] = 1

        # Plot heatmap
        im = ax.imshow(
            np.log10(heatmap.T),
            origin="lower",
            aspect="auto",
            extent=[0, global_axis_max, 0, global_axis_max],
            cmap="inferno",
            vmin=0,
            vmax=np.log10(global_max_count),  # Normalize across all models
        )

        ax.set_xlabel("Predictive Uncertainty")
        ax.set_ylabel("Prediction Error")

        ax.set_title(f"{model_name}\nAvg Correlation: {avg_corr:.2f}")
        # Add diagonal line
        ax.plot(
            [0, global_axis_max],
            [0, global_axis_max],
            color="white",
            linestyle="--",
            linewidth=1,
        )

    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.08, right=0.9, hspace=0.2)

    # Shared colorbar
    fig.colorbar(im, cax=cbar_ax, label=r"$\log_{10}$(Counts)")

    plt.suptitle("Comparative Error Correlation Heatmaps Across Models", y=0.98)

    if save:
        save_plot(fig, "uncertainty_error_corr_comparison.png", config)

    plt.close()


def plot_comparative_dynamic_correlation_heatmaps(
    gradients: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    avg_correlations: dict[str, float],
    max_grad: dict[str, float],
    max_err: dict[str, float],
    max_count: dict[str, float],
    config: dict,
    save: bool = True,
) -> None:
    """
    Plot comparative heatmaps of correlation between gradient and prediction errors
    for multiple surrogate models.

    Args:
        gradients (dict[str, np.ndarray]): Dictionary of gradients from the ensemble of models.
        errors (dict[str, np.ndarray]): Dictionary of prediction errors.
        avg_correlations (dict[str, float]): Dictionary of average correlations between gradients and prediction errors.
        max_grad (dict[str, float]): Dictionary of maximum gradient values for axis scaling across models.
        max_err (dict[str, float]): Dictionary of maximum error values for axis scaling across models.
        max_count (dict[str, float]): Dictionary of maximum count values for heatmap normalization across models.
        config (dict): Configuration dictionary.
        save (bool, optional): Whether to save the plot. Defaults to True.

    Returns:
        None
    """
    # Number of models
    model_names = list(gradients.keys())
    num_models = len(model_names)

    # Determine global max for axis scaling
    global_x_max = max(max_grad.values())
    global_y_max = max(max_err.values())

    # Determine global max for heatmap normalization
    global_max_count = max(max_count.values())

    # Create subplots, one row per model
    fig, axes = plt.subplots(
        num_models, 1, figsize=(8, 6 * num_models), sharex=False, sharey=False
    )

    # Adjust layout for shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        grad_model = gradients[model_name].flatten()
        errors_model = errors[model_name].flatten()
        avg_corr = avg_correlations[model_name]

        # Re-bin the data using the global axis limits
        heatmap, xedges, yedges = np.histogram2d(
            grad_model,
            errors_model,
            bins=100,
            range=[[0, global_x_max], [0, global_y_max]],
        )

        # Set all fields with 0 counts to 1 count for log scale plotting
        heatmap[heatmap == 0] = 1

        # Plot heatmap
        im = ax.imshow(
            np.log10(heatmap.T),
            origin="lower",
            aspect="auto",
            extent=[0, global_x_max, 0, global_y_max],
            cmap="inferno",
            vmin=0,
            vmax=np.log10(global_max_count),  # Normalize across all models
        )

        ax.set_xlabel("Absolute Gradient (Normalized)")
        ax.set_ylabel("Absolute Prediction Error")

        ax.set_title(f"{model_name}\nAvg Correlation: {avg_corr:.2f}")

    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.08, right=0.9, hspace=0.2)

    # Shared colorbar
    fig.colorbar(im, cax=cbar_ax, label=r"$\log_{10}$(Counts)")

    plt.suptitle(
        "Comparative Gradient vs. Prediction Error Heatmaps Across Models", y=0.98
    )

    if save:
        save_plot(fig, "gradients_error_corr_comparison.png", config)

    plt.show()
    plt.close()


def get_custom_palette(num_colors):
    """
    Returns a list of colors sampled from a custom color palette.

    Args:
        num_colors (int): The number of colors needed.

    Returns:
        list: A list of RGBA color tuples.
    """
    # Define your custom color palette as a list of hex color codes
    # custom_palette = [
    #     "#1f77b4",  # blue
    #     "#ff7f0e",  # orange
    #     "#2ca02c",  # green
    #     "#d62728",  # red
    #     "#9467bd",  # purple
    #     "#8c564b",  # brown
    #     "#e377c2",  # pink
    #     "#7f7f7f",  # gray
    #     "#bcbd22",  # yellow
    #     "#17becf",  # teal
    # ]

    custom_palette = [
        "#10002B",  # Darkest violet
        "#3A1A5A",  # Dark purple (slightly brighter)
        "#6A1C75",  # Deep magenta (slightly brighter)
        "#8A3065",  # Magenta with a hint of red
        "#B73779",  # Deep reddish-pink
        "#DA5F4D",  # Dark red-orange
        "#F48842",  # Bright orange
        "#FBC200",  # Yellow-orange (slightly more yellow)
        "#C0CD32",  # Yellow-green (more yellowish)
        "#D2B48C",  # Tan (brownish)
    ]

    # Create a custom colormap from the palette
    custom_cmap = ListedColormap(custom_palette)

    # Sample colors from your custom colormap
    colors = custom_cmap(np.linspace(0, 1, num_colors))

    return colors


def int_ext_sparse(all_metrics: dict, config: dict) -> None:
    """
    Function to make one comparative plot of the interpolation, extrapolation, and sparse training errors.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    surrogates = list(all_metrics.keys())

    # Prepare the data for each modality
    interpolation_metrics_list = []
    interpolation_errors_list = []

    extrapolation_metrics_list = []
    extrapolation_errors_list = []

    sparse_metrics_list = []
    sparse_errors_list = []

    for surrogate in surrogates:
        # Interpolation data
        if "interpolation" in all_metrics[surrogate]:
            interpolation_metrics_list.append(
                all_metrics[surrogate]["interpolation"]["intervals"]
            )
            interpolation_errors_list.append(
                all_metrics[surrogate]["interpolation"]["model_errors"]
            )

        # Extrapolation data
        if "extrapolation" in all_metrics[surrogate]:
            extrapolation_metrics_list.append(
                all_metrics[surrogate]["extrapolation"]["cutoffs"]
            )
            extrapolation_errors_list.append(
                all_metrics[surrogate]["extrapolation"]["model_errors"]
            )

        # Sparse training data
        if "sparse" in all_metrics[surrogate]:
            sparse_metrics_list.append(
                all_metrics[surrogate]["sparse"]["n_train_samples"]
            )
            sparse_errors_list.append(all_metrics[surrogate]["sparse"]["model_errors"])

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(surrogates)))

    # Interpolation subplot
    for i, surrogate in enumerate(surrogates):
        if i < len(interpolation_metrics_list):
            axes[0].scatter(
                interpolation_metrics_list[i],
                interpolation_errors_list[i],
                label=surrogate,
                color=colors[i],
            )
            axes[0].plot(
                interpolation_metrics_list[i],
                interpolation_errors_list[i],
                color=colors[i],
                alpha=0.5,
                linestyle="-",
            )
    axes[0].set_xlabel("Interpolation Intervals")
    axes[0].set_title("Interpolation")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="major", linestyle="--", linewidth=0.5, axis="y")

    # Extrapolation subplot
    for i, surrogate in enumerate(surrogates):
        if i < len(extrapolation_metrics_list):
            axes[1].scatter(
                extrapolation_metrics_list[i],
                extrapolation_errors_list[i],
                label=surrogate,
                color=colors[i],
            )
            axes[1].plot(
                extrapolation_metrics_list[i],
                extrapolation_errors_list[i],
                color=colors[i],
                alpha=0.5,
                linestyle="-",
            )
    axes[1].set_xlabel("Extrapolation Cutoffs")
    axes[1].set_title("Extrapolation")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="major", linestyle="--", linewidth=0.5, axis="y")

    # Sparse training subplot
    for i, surrogate in enumerate(surrogates):
        if i < len(sparse_metrics_list):
            axes[2].scatter(
                sparse_metrics_list[i],
                sparse_errors_list[i],
                label=surrogate,
                color=colors[i],
            )
            axes[2].plot(
                sparse_metrics_list[i],
                sparse_errors_list[i],
                color=colors[i],
                alpha=0.5,
                linestyle="-",
            )
    axes[2].set_xlabel("Number of Training Samples (log scale)")
    axes[2].set_title("Sparse Training")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].grid(True, which="major", linestyle="--", linewidth=0.5, axis="y")

    # Set the x-ticks to all values in sparse_metrics_list
    axes[2].set_xticks(sparse_metrics_list[0])
    axes[2].get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Get current y-axis min and max values
    y_min, y_max = axes[0].get_ylim()

    # Ensure y-axis max does not exceed 1000
    if y_max > 1e3:
        axes[0].set_ylim(y_min, 1e3)

    # Set the ylabel on the leftmost subplot
    axes[0].set_ylabel("Mean Absolute Error")

    # Add the legend to the first plot only
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", fontsize="small")

    # Set the overall title
    fig.suptitle("Model Errors for Different Modalities", fontsize=16)

    # Adjust layout to reduce whitespace and improve spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    save_plot(fig, "generalization_error_comparison.png", config)

    plt.close()


def rel_errors_and_uq(
    metrics: dict[str, dict], config: dict, save: bool = True
) -> None:
    """
    Create a figure with two subplots: relative errors over time and uncertainty over time for different surrogate models.

    Args:
        metrics (dict): Dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool): Whether to save the plot.

    Returns:
        None
    """
    # Prepare data for relative errors plot
    mean_errors = {}
    median_errors = {}
    uncertainties = {}
    absolute_errors = {}
    timesteps = None

    for surrogate, surrogate_metrics in metrics.items():
        relative_error_model = surrogate_metrics["accuracy"].get("relative_errors")
        if relative_error_model is not None:
            mean_errors[surrogate] = np.mean(relative_error_model, axis=(0, 2))
            median_errors[surrogate] = np.median(relative_error_model, axis=(0, 2))

        uncertainties[surrogate] = surrogate_metrics["UQ"]["pred_uncertainty"]
        absolute_errors[surrogate] = surrogate_metrics["accuracy"]["absolute_errors"]
        timesteps = surrogate_metrics["timesteps"]

    # Prepare data for uncertainty plot
    pred_unc_time = {
        surrogate: np.mean(uncertainty, axis=(0, 2))
        for surrogate, uncertainty in uncertainties.items()
    }

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Relative Errors
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(mean_errors)))
    linestyles = ["-", "--"]

    for i, surrogate in enumerate(mean_errors.keys()):
        mean = np.mean(mean_errors[surrogate])
        mean_label = f"{surrogate} Mean = {mean*100:.2f} %"
        ax1.plot(
            timesteps,
            mean_errors[surrogate],
            label=mean_label,
            color=colors[i],
            linestyle=linestyles[0],
        )
        median = np.mean(median_errors[surrogate])
        median_label = f"{surrogate} Median = {median*100:.2f} %"
        ax1.plot(
            timesteps,
            median_errors[surrogate],
            label=median_label,
            color=colors[i],
            linestyle=linestyles[1],
        )

    ax1.set_xlabel("Time")
    ax1.set_xlim(timesteps[0], timesteps[-1])
    # Temp!
    # ax1.set_ylim(3e-4, 1)
    ax1.set_ylabel("Relative Error")
    ax1.set_yscale("log")
    ax1.set_title("Comparison of Relative Errors Over Time")
    ax1.legend(loc="best")

    # Plot 2: Uncertainty over Time
    for i, surrogate in enumerate(pred_unc_time.keys()):
        avg_uncertainty = np.mean(pred_unc_time[surrogate])
        uq_label = f"{surrogate} uncertainty (mean: {avg_uncertainty:.2e})"
        ax2.plot(
            timesteps,
            pred_unc_time[surrogate],
            label=uq_label,
            linestyle="-",
            color=colors[i],
        )

        abs_errors_time = np.mean(absolute_errors[surrogate], axis=(0, 2))
        abs_error_avg = np.mean(abs_errors_time)
        err_label = f"{surrogate} abs. error (mean: {abs_error_avg:.2e})"
        ax2.plot(
            timesteps,
            abs_errors_time,
            label=err_label,
            color=colors[i],
            linestyle="--",
        )

    ax2.set_xlabel("Time")
    ax2.set_xlim(timesteps[0], timesteps[-1])
    # Temp!
    # ax2.set_ylim(0, 0.04)
    ax2.set_ylabel("Uncertainty/Absolute Error")
    ax2.set_title("Comparison of Predictive Uncertainty Over Time")
    ax2.legend(loc="best")

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Save plot if required
    if save and config:
        save_plot(fig, "combined_rel_errors_and_uq.png", config)

    plt.close()
