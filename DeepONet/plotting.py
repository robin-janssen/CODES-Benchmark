from __future__ import annotations

import matplotlib.pyplot as plt
from typing import Optional
from itertools import cycle
import numpy as np
from DeepONet.utils import create_date_based_directory, save_plot_counter


def plot_functions_and_antiderivatives(
    data_poly: list,
    data_GRF: list,
    data_sine: list,
    sensor_points: np.array,
    num_samples_to_plot: int = 3,
) -> None:
    """
    Plot some samples of the generated data (poly, GRF, sine) and their antiderivatives.

    :param data_poly: List of tuples containing the polynomial data and its antiderivative.
    :param data_GRF: List of tuples containing the GRF data and its antiderivative.
    :param data_sine: List of tuples containing the sine data and its antiderivative.
    Note: Each tuple contains two numpy arrays: the function and its antiderivative evaluated at sensor points.
    :param sensor_points: Array of sensor locations in the domain.
    :param num_samples_to_plot: Number of samples to plot.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Some samples of the data")

    colors = cycle(plt.cm.tab10.colors)  # Color cycle from a matplotlib colormap

    for i in range(num_samples_to_plot):
        color = next(colors)
        axs[0].plot(sensor_points, data_poly[i][0], label=f"poly_{i+1}", color=color)
        axs[0].plot(
            sensor_points,
            data_poly[i][1],
            label=f"poly_int_{i+1}",
            color=color,
            linestyle="--",
        )
        axs[1].plot(
            sensor_points,
            data_GRF[i][0],
            label=f"GRF_{i+1}",
            color=color,
            linestyle="-",
        )
        axs[1].plot(
            sensor_points,
            data_GRF[i][1],
            label=f"GRF_int_{i+1}",
            color=color,
            linestyle="--",
        )
        axs[2].plot(
            sensor_points,
            data_sine[i][0],
            label=f"sine_{i+1}",
            color=color,
            linestyle="-",
        )
        axs[2].plot(
            sensor_points,
            data_sine[i][1],
            label=f"sine_int_{i+1}",
            color=color,
            linestyle="--",
        )

    axs[0].set_title("Polynomial data")
    axs[1].set_title("Gaussian random field data")
    axs[2].set_title("Sine data")

    for ax in axs:
        ax.set_xlabel("Domain")
        ax.set_ylabel("Function Value")

    plt.tight_layout()
    plt.show()


def plot_functions_only(
    data: list, sensor_points: np.array, num_samples_to_plot: int = 3
) -> None:
    """
    Plot some samples of the data

    :param data: list of tuples containing the function and its antiderivative.
    Note: Each tuple contains two numpy arrays: the function and its antiderivative evaluated at sensor points.
    :param sensor_points: Array of sensor locations in the domain.
    :param num_samples_to_plot: Number of samples to plot.

    """
    plt.figure(figsize=(6, 4))
    plt.title("Function Coverage")

    for i in range(num_samples_to_plot):
        plt.plot(sensor_points, data[i, :, 0], color="blue", alpha=0.2)  # Low opacity

    plt.xlabel("Domain")
    plt.ylabel("Function Value")
    plt.tight_layout()
    plt.show()


def surface_plot(
    sensor_locations: np.array,
    timesteps: np.array,
    functions: np.array,
    num_samples_to_plot: int = 3,
    predictions: np.array = None,
    title: str = "Data Visualization",
) -> None:
    """
    Plot the evolution of functions over time using a 3D surface plot.
    If predictions are provided, plot the ground truth, predictions, and error in separate subplots.

    :param sensor_locations: Array of sensor locations.
    :param timesteps: Array of timesteps.
    :param functions: 3D array of ground truth function values - shape [N_samples, len(sensor_locations), len(timesteps)].
    :param num_samples_to_plot: Number of samples to plot.
    :param title: Title of the plot.
    :param predictions: 3D array of predicted function values (optional).
    """

    X, Y = np.meshgrid(sensor_locations, timesteps)
    total_cols = num_samples_to_plot

    if predictions is None:
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(title)
        for i in range(num_samples_to_plot):
            Z = functions[i].T
            plot_single_surface(fig, 1, i, 1, total_cols, X, Y, Z, "Data", i)
    else:
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(title)
        for i in range(num_samples_to_plot):
            # Plot ground truth
            Z_gt = functions[i].T
            plot_single_surface(fig, 1, i, 3, total_cols, X, Y, Z_gt, "Ground Truth", i)

            # Plot predictions
            Z_pred = predictions[i].T
            plot_single_surface(fig, 2, i, 3, total_cols, X, Y, Z_pred, "Prediction", i)

            # Plot error
            Z_err = Z_gt - Z_pred
            plot_single_surface(fig, 3, i, 3, total_cols, X, Y, Z_err, "Error", i)

    plt.gcf().subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.show()


def plot_single_surface(
    fig: plt.Figure,
    row: int,
    col: int,
    total_rows: int,
    total_cols: int,
    X: np.array,
    Y: np.array,
    Z: np.array,
    title_prefix: str,
    sample_number: int,
) -> None:
    """
    Plot a single surface in a subplot.

    :param fig: The figure object to which the subplot is added.
    :param row: The row number of the subplot.
    :param col: The column number of the subplot.
    :param total_rows: Total number of rows in the subplot grid.
    :param total_cols: Total number of columns in the subplot grid.
    :param X: X-coordinates for the meshgrid (shape [len(sensor_points), len(timesteps)]).
    :param Y: Y-coordinates for the meshgrid (shape [len(sensor_points), len(timesteps)]).
    :param Z: Z-coordinates (function values) for the plot (shape [len(sensor_points), len(timesteps)]).
    :param title_prefix: Prefix for the subplot title.
    :param sample_number: The sample number (for the title).
    """
    if total_rows == 1:
        # When there's only one row, the index is the sample number plus one
        index = sample_number + 1
    else:
        # For multiple rows, calculate the index based on row and column
        index = (row - 1) * total_cols + col + 1

    ax = fig.add_subplot(total_rows, total_cols, index, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Sensor Location")
    ax.set_ylabel("Time")
    ax.set_zlabel("Value")
    ax.set_title(f"{title_prefix} Sample {sample_number + 1}")


def heatmap_plot(
    sensor_locations: np.array,
    timesteps: np.array,
    functions: np.array,
    num_samples_to_plot: int = 3,
    predictions: np.array = None,
    title: str = "Data Visualization",
) -> None:
    """
    Plot the evolution of functions over time using heat maps with a common colorbar.

    :param sensor_locations: Array of sensor locations.
    :param timesteps: Array of timesteps.
    :param functions: 3D array of ground truth function values.
    :param num_samples_to_plot: Number of samples to plot.
    :param title: Title of the plot.
    :param predictions: 3D array of predicted function values (optional).
    """
    figsize = (10, 3) if predictions is None else (12, 8)
    fig, axs = plt.subplots(
        3 if predictions is not None else 1,
        num_samples_to_plot,
        figsize=figsize,
        squeeze=False,
        layout="compressed",
    )
    fig.suptitle(title)
    functions = functions[:num_samples_to_plot].transpose(0, 2, 1)
    predictions = (
        predictions[:num_samples_to_plot].transpose(0, 2, 1)
        if predictions is not None
        else None
    )

    # Determine common vmin and vmax for color scaling
    vmin = min(
        functions.min(),
        predictions.min() if predictions is not None else functions.min(),
    )
    vmax = max(
        functions.max(),
        predictions.max() if predictions is not None else functions.max(),
    )

    for i in range(num_samples_to_plot):
        # Plot ground truth
        im = axs[0, i].imshow(
            functions[i],
            aspect="equal",
            origin="lower",
            extent=[
                sensor_locations[0],
                sensor_locations[-1],
                timesteps[0],
                timesteps[-1],
            ],
            vmin=vmin,
            vmax=vmax,
        )
        axs[0, i].set_title(f"Ground Truth Sample {i+1}")
        axs[0, i].set_xlabel("Sensor Location")
        axs[0, i].set_ylabel("Time")

        if predictions is not None:
            # Plot predictions
            axs[1, i].imshow(
                predictions[i],
                aspect="equal",
                origin="lower",
                extent=[
                    sensor_locations[0],
                    sensor_locations[-1],
                    timesteps[0],
                    timesteps[-1],
                ],
                vmin=vmin,
                vmax=vmax,
            )
            axs[1, i].set_title(f"Prediction Sample {i+1}")
            axs[1, i].set_xlabel("Sensor Location")
            axs[1, i].set_ylabel("Time")

            # Plot error
            error = functions[i] - predictions[i]
            axs[2, i].imshow(
                error,
                aspect="equal",
                origin="lower",
                extent=[
                    sensor_locations[0],
                    sensor_locations[-1],
                    timesteps[0],
                    timesteps[-1],
                ],
                vmin=vmin,
                vmax=vmax,
            )
            axs[2, i].set_title(f"Error Sample {i+1}")
            axs[2, i].set_xlabel("Sensor Location")
            axs[2, i].set_ylabel("Time")

    # Add a common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.show()


def heatmap_plot_errors(
    sensor_locations: np.array,
    timesteps: np.array,
    errors: list,
    num_samples_to_plot: int = 3,
    title: str = "Model Comparison via Error Visualization",
) -> None:
    """
    Plot the error evolution of different models over time using heat maps with a common colorbar.

    :param sensor_locations: Array of sensor locations.
    :param timesteps: Array of timesteps.
    :param errors: List of 3D arrays of error values for different models.
    :param num_samples_to_plot: Number of samples to plot.
    :param title: Title of the plot.
    """
    num_models = len(errors)  # Determine number of models based on the list length
    figsize = (12, 3 * num_models)  # Adjust figure size based on the number of models
    fig, axs = plt.subplots(
        num_models,
        num_samples_to_plot,
        figsize=figsize,
        squeeze=False,
    )
    fig.suptitle(title)

    # Determine common vmin and vmax for color scaling across all models
    vmin = min(error.min() for error in errors)
    vmax = max(error.max() for error in errors)

    for model_idx, model_errors in enumerate(errors):
        model_errors = model_errors[:num_samples_to_plot].transpose(
            0, 2, 1
        )  # Adjust data shape

        for sample_idx in range(num_samples_to_plot):
            im = axs[model_idx, sample_idx].imshow(
                model_errors[sample_idx],
                aspect="equal",
                origin="lower",
                extent=[
                    sensor_locations[0],
                    sensor_locations[-1],
                    timesteps[0],
                    timesteps[-1],
                ],
                vmin=vmin,
                vmax=vmax,
            )
            axs[model_idx, sample_idx].set_title(
                f"Model {model_idx+1} Error {sample_idx+1}"
            )
            axs[model_idx, sample_idx].set_xlabel("Sensor Location")
            axs[model_idx, sample_idx].set_ylabel("Time")

    # Adjust colorbar to fit the height of the plot dynamically
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust as necessary
    fig.colorbar(im, cax=cbar_ax)

    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to prevent overlap
    plt.show()


def plot_losses(
    loss_histories: tuple[np.array, ...],
    labels: tuple[str, ...],
    title: str = "Losses",
    save: bool = False,
) -> None:
    """
    Plot the loss trajectories for the training of multiple models.

    :param loss_histories: List of loss history arrays.
    :param labels: List of labels for each loss history.
    :param title: Title of the plot.
    :param store_plot: Whether to store the plot as an image file.
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

    if save:
        filename = "losses.png"
        directory = create_date_based_directory(subfolder="plots")
        filepath = save_plot_counter(filename, directory)
        plt.savefig(filepath)
        print(f"Plot saved as: {filepath}")

    plt.show()


def plot_results(
    title,
    sensor_locations,
    function_values,
    ground_truth,
    y_values,
    predictions,
    num_samples=3,
):
    colors = cycle(plt.cm.tab10.colors)  # Color cycle from a matplotlib colormap

    plt.figure(figsize=(12, 6))
    plt.suptitle(title)

    # First subplot: Input function, Ground Truth, and Predictions
    plt.subplot(1, 2, 1)
    for i in range(num_samples):
        color = next(colors)
        # plt.plot(sensor_locations, function_values[i], label=f"Input Function {i+1}", color=color, linestyle='-', marker='o')
        plt.plot(
            sensor_locations,
            ground_truth[i],
            label=f"Ground Truth {i+1}",
            color=color,
            linestyle="--",
        )
        plt.plot(
            y_values,
            predictions[i],
            label=f"DeepONet Prediction {i+1}",
            color=color,
            linestyle=":",
            marker=".",
        )
    plt.xlabel("Domain")
    plt.ylabel("Function Value")
    plt.title("Function, Ground Truth, and Predictions")
    plt.legend()

    # Second subplot: Error
    plt.subplot(1, 2, 2)
    for i in range(num_samples):
        color = next(colors)
        error = ground_truth[i] - predictions[i]
        plt.plot(y_values, error, label=f"Error {i+1}", color=color)
    plt.xlabel("Domain")
    plt.ylabel("Error")
    plt.title("Prediction Error")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_chemical_examples(
    data: np.array,
    names: list[str] | None = None,
    num_chemicals: int | None = None,
    save: bool = False,
    title: str = "Chemical Examples",
) -> None:
    """Creates four exemplary plots, displaying the amount of each chemical over time.

    :param data: 3D numpy array with the chemical data.
    :param names: Optional list of strings with the names of the chemicals.
    :param num_chemicals: Number of chemicals to display. If None, all chemicals are displayed.
    :param save: Whether to save the plot as an image file.
    :param title: Title of the plot.
    :return: None
    """
    if num_chemicals is None:
        num_chemicals = data.shape[2]
    if names is None:
        names = [f"Chem {i+1}" for i in range(num_chemicals)]
    plt.subplots(2, 2, figsize=(10, 10))
    plt.suptitle(title)
    for j in range(4):
        for i in range(num_chemicals):
            plt.subplot(2, 2, j + 1)
            plt.plot(data[j, :, i], label=names[i])
        plt.xlabel("Timestep")
        plt.ylabel("Amount")
        # plt.legend()

    if save:
        filename = "chemical_examples.png"
        directory = create_date_based_directory(subfolder="plots")
        filepath = save_plot_counter(filename, directory)
        plt.savefig(filepath)
        print(f"Plot saved as: {filepath}")

    plt.show()


def plot_chemicals_comparative(
    data: np.array,
    names: list[str] | None = None,
    num_chemicals: int | None = None,
    save: bool = False,
    title: str = "Chemical Comparison",
) -> None:
    """Creates four plots, displaying the evolution of four chemicals over time.

    :param data: 3D numpy array with the chemical data.
    :param names: Optional list of strings with the names of the chemicals.
    :param num_chemicals: Number of chemicals to display. If None, all chemicals are displayed.
    :param save: Whether to save the plot as an image file.
    :param title: Title of the plot.
    :return: None
    """
    if num_chemicals is None:
        num_chemicals = data.shape[2]
    if names is None:
        names = [f"Chem {i+1}" for i in range(data.shape[2])]
    plt.subplots(2, 2, figsize=(10, 10))
    plt.suptitle(title)
    for j in range(4):
        plt.subplot(2, 2, j + 1)
        for i in range(num_chemicals):
            plt.plot(data[i, :, 3 * j], label=f"{names[3 * j]}")
        plt.xlabel("Timestep")
        plt.ylabel("Amount")
        # plt.legend()
    if save:
        filename = "chemical_comparison.png"
        directory = create_date_based_directory(subfolder="plots")
        filepath = save_plot_counter(filename, directory)
        plt.savefig(filepath)
        print(f"Plot saved as: {filepath}")

    plt.show()


def plot_chemical_results_2(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    names: list[str],
    num_chemicals: int | None = None,
) -> None:
    """
    Plot the results of the chemical predictions

    :param predictions: 3d numpy array of shape (num_samples, num_timesteps, num_chemicals)
    :param ground_truth: 3d numpy array of shape (num_samples, num_timesteps, num_chemicals)
    :param names: list of strings with the chemical names.
    :param num_chemicals: number of chemicals to plot. None to plot all.
    """
    # Generate colors
    if num_chemicals is None:
        num_chemicals = predictions.shape[2]
    c = plt.cm.viridis(np.linspace(0, 1, num_chemicals))

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle("Chemical predictions")
    for i in range(4):
        for j in range(num_chemicals):
            ax[i % 2, i // 2].plot(
                predictions[i, :, j], label=f"P {names[j]}", linestyle="--", color=c[j]
            )
            ax[i % 2, i // 2].plot(
                ground_truth[i, :, j], label=f"GT {names[j]}", linestyle="-", color=c[j]
            )
            ax[i % 2, i // 2].set_title("Example " + str(i + 1))
    # Add a common legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.show()


# def plot_chemical_results(
#     predictions: np.ndarray,
#     ground_truth: np.ndarray,
#     names: list[str],
#     num_chemicals: int | None = None,
# ) -> None:
#     """
#     Plot the results of the chemical predictions.

#     :param predictions: 3d numpy array of shape (num_samples, num_timesteps, num_chemicals)
#     :param ground_truth: 3d numpy array of shape (num_samples, num_timesteps, num_chemicals)
#     :param names: list of strings with the chemical names.
#     :param num_chemicals: number of chemicals to plot. None to plot all.
#     """
#     # Generate colors
#     if num_chemicals is None:
#         num_chemicals = predictions.shape[2]
#     c = plt.cm.viridis(np.linspace(0, 1, num_chemicals))

#     fig, ax = plt.subplots(2, 2, figsize=(10, 8))
#     plt.suptitle("Chemical predictions")
#     for i in range(4):
#         for j in range(num_chemicals):
#             ax[i % 2, i // 2].plot(
#                 predictions[i, :, j], label=f"P {names[j]}", linestyle="--", color=c[j]
#             )
#             ax[i % 2, i // 2].plot(
#                 ground_truth[i, :, j], label=f"GT {names[j]}", linestyle="-", color=c[j]
#             )
#             ax[i % 2, i // 2].set_title("Example " + str(i + 1))

#     # Add a common legend outside the plots
#     handles, labels = ax[0, 0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 0.5))

#     # Adjust layout to make room for the legend
#     plt.tight_layout(rect=[0, 0, 0.9, 1])

#     plt.show()


def plot_chemical_results(
    predictions: np.ndarray | tuple[np.ndarray, ...],
    ground_truth: np.ndarray,
    names: list[str] | None = None,
    model_names: str | tuple[str, ...] = "Model",
    num_chemicals: int | None = None,
    save: bool = False,
    title: str = "Chemical Predictions",
) -> None:
    """
    Plot the results of the chemical predictions.

    :param predictions: Either a 3D numpy array of shape (num_samples, num_timesteps, num_chemicals) for single predictions,
                        or a tuple of such arrays for multiple predictions.
    :param ground_truth: 3D numpy array of shape (num_samples, num_timesteps, num_chemicals)
    :param names: list of strings with the chemical names.
    :param model_names: string or tuple of strings for the prediction labels.
    :param num_chemicals: number of chemicals to plot. None to plot all.
    :param save: whether to save the plot as an image file.
    """
    # Check if predictions is a tuple of numpy arrays (multiple predictions) or a single numpy array
    if not isinstance(predictions, tuple):
        predictions = (predictions,)  # Make it a tuple to generalize the plotting logic
    if isinstance(model_names, str):
        model_names = (model_names,)  # Make it a tuple for uniform handling

    # Determine number of chemicals to plot
    if num_chemicals is None:
        num_chemicals = ground_truth.shape[2]

    # Generate colors
    c = plt.cm.viridis(np.linspace(0, 1, num_chemicals))

    alphas = np.linspace(0.5, 1, len(predictions))

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    plt.suptitle(title)
    for i in range(4):
        for j in range(num_chemicals):
            # Plot each set of predictions
            for pred_idx, pred_set in enumerate(predictions):
                pred_label = (
                    f"{model_names[pred_idx]} {names[j]}"
                    if names is not None
                    else f"{model_names[pred_idx]}"
                )
                ax[i % 2, i // 2].plot(
                    pred_set[i, :, j],
                    label=pred_label,
                    linestyle="--",
                    color=c[j],
                    alpha=alphas[pred_idx],
                )
            # Plot ground truth
            gt_label = f"GT {names[j]}" if names is not None else "Ground Truth"
            ax[i % 2, i // 2].plot(
                ground_truth[i, :, j], label=gt_label, linestyle="-", color=c[j]
            )
            ax[i % 2, i // 2].set_title(f"Example {i + 1}")

    # Adjust legend and layout
    if names is not None:
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save:
        filename = "chemical_predictions.png"
        directory = create_date_based_directory(subfolder="plots")
        filepath = save_plot_counter(filename, directory)
        plt.savefig(filepath)
        print(f"Plot saved as: {filepath}")

    plt.show()


def plot_chemical_results_and_errors(
    predictions: np.ndarray | tuple[np.ndarray, ...],
    ground_truth: np.ndarray,
    names: list[str],
    model_names: str | tuple[str, ...],
    num_chemicals: int | None = None,
) -> None:
    """
    Plot the results of the chemical predictions and their relative errors.

    :param predictions: Either a 3D numpy array or a tuple of 3D numpy arrays (predictions).
    :param ground_truth: 3D numpy array of shape (num_samples, num_timesteps, num_chemicals).
    :param names: list of strings with the chemical names.
    :param model_names: string or tuple of strings for the prediction labels.
    :param num_chemicals: number of chemicals to plot. None to plot all.
    """
    # Ensure predictions and model_names are tuples for uniform handling
    if not isinstance(predictions, tuple):
        predictions = (predictions,)
    if isinstance(model_names, str):
        model_names = (model_names,)

    if num_chemicals is None:
        num_chemicals = ground_truth.shape[2]

    c = plt.cm.viridis(np.linspace(0, 1, num_chemicals))
    alphas = np.linspace(0.2, 1, len(predictions))

    fig, ax = plt.subplots(2, 3, figsize=(12, 9))  # Adjusted for 3 examples and 2 rows
    plt.suptitle("Chemical Predictions and Errors")

    # Plot predictions
    for i in range(3):
        for j in range(num_chemicals):
            for pred_idx, pred_set in enumerate(predictions):
                ax[0, i].plot(
                    pred_set[i, :, j],
                    label=f"{model_names[pred_idx]} {names[j]}",
                    linestyle="-",
                    color=c[j],
                    alpha=alphas[pred_idx],
                    linewidth=1,
                )
            ax[0, i].plot(
                ground_truth[i, :, j],
                label=f"GT {names[j]}",
                linestyle="--",
                color=c[j],
                alpha=1,
            )
            ax[0, i].set_title(f"Example {i + 1}")

    # Plot relative errors
    for i in range(3):
        for j in range(num_chemicals):
            gt = ground_truth[i, :, j]
            for pred_idx, pred_set in enumerate(predictions):
                pred = pred_set[i, :, j]
                rel_error = np.abs(pred - gt) / np.abs(gt)
                ax[1, i].plot(
                    rel_error,
                    label=f"Error {model_names[pred_idx]} {names[j]}",
                    linestyle="-",
                    color=c[j],
                    alpha=alphas[pred_idx],
                )
            ax[1, i].set_title(f"Error Example {i + 1}")

    # Adjust legend and layout
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


def plot_relative_errors_over_time(
    relative_errors: np.ndarray, title: str, save: bool = False
) -> None:
    """
    Plot the mean and median relative errors over time with shaded regions for
    the 50th, 90th, and 99th percentiles.

    :param relative_errors: 3D numpy array of shape [num_samples, timesteps, num_chemicals].
    :param title: Title of the plot.
    :param save: Whether to save the plot as a file.
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

    # plt.ylim(1e-5, 1)
    plt.yscale("log")
    plt.xlabel("Timestep")
    plt.ylabel("Relative Error (Log Scale)")
    plt.title(title)
    plt.legend()

    if save:
        filename = "relative_errors.png"
        directory = create_date_based_directory(subfolder="plots")
        filepath = save_plot_counter(filename, directory)
        plt.savefig(filepath)
        print(f"Plot saved as: {filepath}")

    plt.show()


def plot_chemical_errors(
    errors: np.ndarray,
    extracted_chemicals: list,
    num_chemicals: int | None = None,
    title: str = "Mean errors for each chemical",
):
    if num_chemicals is None:
        num_chemicals = errors.shape[1]
    plt.figure(figsize=(10, 5))
    for i in range(num_chemicals):
        plt.plot(errors[:, i], label=extracted_chemicals[i])
    plt.yscale("log")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_mass_conservation(ground_truth, masses, num_examples=5):
    # Ensure masses is a numpy array to utilize broadcasting
    masses = np.array(masses)

    plt.figure(figsize=(12, 8))

    # Calculate and plot the total mass for a selection of examples
    for i in range(num_examples):
        # Calculate the total mass for each timestep
        total_mass = np.sum(ground_truth[i] * masses, axis=1)  # Sum over chemicals

        # Plot
        plt.plot(total_mass, label=f"Example {i+1}")

    plt.xlabel("Timestep")
    plt.ylabel("Total Mass")
    plt.title("Total Mass Conservation Over Time")
    plt.legend()
    plt.show()


def visualise_deep_ensemble(
    predictions_list: list[np.ndarray],
    ground_truth: np.ndarray,
    num_chemicals: int,
    chemical_names: list[str],
    save: bool = False,
):
    """
    Visualize the predictions of a deep ensemble and the ground truth.

    :param predictions_list: List of arrays with shape [N_datapoints, N_timesteps, N_chemicals]
    :param ground_truth: Array of shape [N_datapoints, N_timesteps, N_chemicals]
    :param num_chemicals: Number of chemicals to plot
    :param chemical_names: List of chemical names
    :param save: Whether to save the plot as an image file
    """
    # Ensure num_chemicals does not exceed the size of the third dimension
    num_chemicals = min(num_chemicals, ground_truth.shape[2])

    # Calculate mean and standard deviation of predictions
    predictions_stack = np.stack(predictions_list, axis=0)
    prediction_mean = np.mean(predictions_stack, axis=0)
    prediction_std = np.std(predictions_stack, axis=0)

    # Generate colors
    colors = plt.cm.viridis(np.linspace(0, 1, num_chemicals))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for datapoint_idx in range(4):  # Assuming four subplots
        ax = axs[datapoint_idx // 2, datapoint_idx % 2]
        for chem_idx in range(num_chemicals):
            idx_factor = 1  # Just to select different chemicals
            chem_idx = chem_idx * idx_factor
            gt = ground_truth[:, :, chem_idx]
            mean = prediction_mean[:, :, chem_idx]
            std = prediction_std[:, :, chem_idx]

            timesteps = np.arange(gt.shape[1])
            ax.plot(
                timesteps,
                gt[datapoint_idx],
                color=colors[chem_idx // idx_factor],
                label=f"GT {chemical_names[chem_idx]}",
            )
            ax.plot(
                timesteps,
                mean[datapoint_idx],
                "--",
                color=colors[chem_idx // idx_factor],
                label=f"Pred Chem {chemical_names[chem_idx]}",
            )

            # Plot standard deviations as shaded areas
            for sigma_multiplier in [1, 2, 3]:  # 1, 2, and 3 standard deviations
                ax.fill_between(
                    timesteps,
                    mean[datapoint_idx] - sigma_multiplier * std[datapoint_idx],
                    mean[datapoint_idx] + sigma_multiplier * std[datapoint_idx],
                    color=colors[chem_idx // idx_factor],
                    alpha=0.5 / sigma_multiplier,
                )

    plt.legend()
    plt.tight_layout()

    if save:
        filename = "deep_ensemble_predictions.png"
        directory = create_date_based_directory(subfolder="plots")
        filepath = save_plot_counter(filename, directory)
        plt.savefig(filepath)
        print(f"Plot saved as: {filepath}")

    plt.show()


def plot_spectrum_predictions(
    predictions: np.ndarray | tuple[np.ndarray, ...],
    ground_truth: np.ndarray,
    timesteps: tuple[float, ...],
    momenta: np.ndarray,
    model_names: str | tuple[str, ...] = "Model",
    save: bool = False,
) -> None:
    """
    Plot the results of the spectral predictions.

    :param predictions: Either a 3D numpy array of shape (num_samples, num_timesteps, num_momenta) for single predictions,
                        or a tuple of such arrays for multiple predictions.
    :param ground_truth: 3D numpy array of shape (num_samples, num_timesteps, num_momenta)
    :param timesteps: Tuple of floats representing the time steps.
    :param momenta: 1D numpy array of momentum values.
    :param model_names: string or tuple of strings for the prediction labels.
    :param save: Whether to save the plot as a file.
    """
    if not isinstance(predictions, tuple):
        predictions = (predictions,)  # Make it a tuple to generalize the plotting logic
    if isinstance(model_names, str):
        model_names = (model_names,)  # Make it a tuple for uniform handling

    # Setup plot
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle("Spectral Time-Evolution Predictions (DeepONet)")

    for i in range(4):
        ax_flat = ax[i % 2, i // 2]
        # Plot each timestep
        for t_idx, time in enumerate(timesteps):
            # gt_label = f"GT t={time:.1f}"
            ax_flat.plot(
                momenta,
                ground_truth[i, t_idx, :],
                # label=gt_label,
                linestyle="-",
                color="tab:blue",
            )

            # Plot each set of predictions
            for pred_idx, pred_set in enumerate(predictions):
                if len(predictions) > 1:
                    pred_label = f"{model_names[pred_idx*3]} t={time:.3f}"
                else:
                    pred_label = f"t={time:.3f}"
                ax_flat.plot(
                    momenta,
                    pred_set[i, t_idx, :],
                    label=pred_label,
                    linestyle="--",
                    color="tab:orange",
                    alpha=0.8,
                )

        ax_flat.set_title(f"Sample {i + 1}")
        ax_flat.set_xlabel(r"$p$")
        ax_flat.set_ylabel(r"$p^2 f$")
        ax_flat.legend()

    if save:
        filename = "spectrum_predictions.png"
        directory = create_date_based_directory(subfolder="plots")
        filepath = save_plot_counter(filename, directory)
        plt.savefig(filepath)
        print(f"Plot saved as: {filepath}")

    plt.tight_layout()
    plt.show()


def compare_datasets_histogram(dataset1, dataset2, dataset3, save=False):
    """
    Compare three datasets using histograms for each of their columns.

    :param dataset1: numpy array of shape [num_samples, 10]
    :param dataset2: numpy array of shape [num_samples, 10]
    :param dataset3: numpy array of shape [num_samples, 10]
    :param save: whether to save the plot as a file
    """
    num_columns = dataset1.shape[1]
    fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 30))

    # Iterate over each quantity (column)
    for i in range(num_columns):
        ax = axes[i]
        # Find the range for the histograms
        combined_data = np.concatenate((dataset1[:, i], dataset2[:, i], dataset3[:, i]))

        data_range = (np.min(combined_data), np.max(combined_data))
        data_range = (1e-10, data_range[1]) if data_range[0] <= 0 else data_range

        logbins = np.logspace(np.log10(data_range[0]), np.log10(data_range[1]), 50)

        # Plot histograms for each dataset in the current subplot
        # Make the xscale logarithmic for better visualization
        ax.hist(
            dataset1[:, i],
            bins=logbins,
            range=data_range,
            alpha=0.3,
            label="Train data (1e6)",
            density=True,
        )
        ax.hist(
            dataset2[:, i],
            bins=logbins,
            range=data_range,
            alpha=0.3,
            label="Test data (3e5)",
            density=True,
        )
        ax.hist(
            dataset3[:, i],
            bins=logbins,
            range=data_range,
            alpha=0.3,
            label="Biggest dataset (13.4e6)",
            density=True,
        )

        ax.set_xscale("log")

        ax.set_title(f"Quantity {i + 1}")
        ax.legend()

    plt.tight_layout()

    if save:
        filename = "branca_dataset_comparison.png"
        directory = create_date_based_directory(subfolder="plots")
        filepath = save_plot_counter(filename, directory)
        plt.savefig(filepath)
        print(f"Plot saved as: {filepath}")

    plt.show()


def plot_generalization_errors(
    metrics: np.array,
    model_errors: np.array,
    title: Optional[str] = None,
    interpolate: bool = True,
    save: bool = False,
) -> None:
    """
    Plot the interpolation or extrapolation errors of a model.

    Args:
        metrics: Numpy array containing the generalization metric (interpolation interval or extrapolation cutoff) values.
        model_errors: Numpy array containing the model errors.
        title: Optional title for the plot.
        interpolate: Whether the plot comes from interpolation or extrapolation.
        save: Whether to save the plot as a file.

    Returns:
        None
    """
    plt.scatter(metrics, model_errors)
    xlabel = "Interpolation Interval" if interpolate else "Extrapolation Cutoff"
    plt.xlabel(xlabel)
    plt.ylabel("Mean Absolute Error")
    plt.yscale("log")
    if title is None:
        title = "Interpolation Errors" if interpolate else "Extrapolation Errors"
    plt.title(title)

    if save:
        filename = (
            "interpolation_errors.png" if interpolate else "extrapolation_errors.png"
        )
        directory = create_date_based_directory(subfolder="plots")
        filepath = save_plot_counter(filename, directory)
        plt.savefig(filepath)
        print(f"Plot saved as: {filepath}")

    plt.show()
