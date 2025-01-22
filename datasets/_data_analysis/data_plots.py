import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(1, "../..")

from codes import get_custom_palette, save_plot


def plot_example_trajectories(
    dataset_name: str,
    data: np.ndarray,
    timesteps: np.ndarray,
    num_chemicals: int = 10,
    labels: list[str] | None = None,
    save: bool = False,
    sample_idx: int = 0,
    log: bool = False,
    quantities_per_plot: int = 6,  # Added parameter to split quantities into subplots
) -> None:
    """
    Plot example trajectories for the dataset.

    Args:
        dataset_name (str): The name of the dataset.
        data (np.ndarray): The data to plot.
        timesteps (np.ndarray): Timesteps array.
        num_chemicals (int, optional): Number of chemicals to plot. Default is 10.
        labels (list, optional): List of labels for the chemicals.
        save (bool, optional): Whether to save the plot as a file.
        sample_idx (int, optional): Index of the sample to plot.
        log (bool, optional): Whether to plot the data in log-space.
        quantities_per_plot (int, optional): Number of quantities to plot per subplot. Default is 6.
    """
    # Cap the number of chemicals at the number of available chemicals
    num_chemicals = min(data.shape[2], num_chemicals)
    data = data[sample_idx, :, :num_chemicals]

    # Define the color palette
    colors = plt.cm.viridis(np.linspace(0, 0.9, quantities_per_plot))

    # Split the quantities into groups of quantities_per_plot
    num_plots = int(np.ceil(num_chemicals / quantities_per_plot))

    # Create subplots with shared x-axis
    fig, axes = plt.subplots(num_plots, 1, figsize=(9, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even if there's only one plot

    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        start_idx = plot_idx * quantities_per_plot
        end_idx = min((plot_idx + 1) * quantities_per_plot, num_chemicals)

        for chem_idx in range(start_idx, end_idx):
            color = colors[chem_idx % quantities_per_plot]
            gt = data[:, chem_idx]

            # Plot ground truth
            ax.plot(
                timesteps,
                gt,
                "-",
                color=color,
                label=(
                    f"Quantity {chem_idx + 1}" if labels is None else labels[chem_idx]
                ),
            )

        # Set labels and title
        ax.set_xlabel("Time")
        ax.set_xlim(timesteps[0], timesteps[-1])
        ylabel = "log(Abundance)" if log else "Abundance"
        ax.set_ylabel(ylabel)

        # Add legend next to the subplot
        if labels is not None:
            ax.legend(
                loc="center left",  # Align the legend to the center of the plot
                bbox_to_anchor=(1.03, 0.5),  # Place legend to the right of the plot
                borderaxespad=0.0,  # Remove padding
            )

    fig.suptitle(f"Example Trajectories for Dataset: {dataset_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the plot if specified
    if save:
        conf = {
            "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
            "verbose": True,
        }
        save_plot(
            plt,
            "example_trajectories.png",
            conf,
            dpi=200,
            base_dir="datasets",
            increase_count=False,
        )

    plt.close()


def plot_example_trajectories_paper(
    dataset_name: str,
    data: np.ndarray,
    timesteps: np.ndarray,
    save: bool = False,
    sample_idx: int = 0,
    labels: list[str] | None = None,
) -> None:
    """
    Plot example trajectories for the dataset with two subplots, one showing 15 chemicals and another showing the remaining.

    Args:
        dataset_name (str): The name of the dataset.
        data (np.ndarray): The data to plot.
        timesteps (np.ndarray): Timesteps array.
        save (bool, optional): Whether to save the plot as a file.
        sample_idx (int, optional): Index of the sample to plot.
        labels (list, optional): List of labels for the chemicals.
    """
    # Ensure we are plotting the correct sample
    data = data[sample_idx]

    # Define the number of chemicals per subplot
    total_chemicals = data.shape[1]
    num_chemicals_subplots = [
        total_chemicals // 2,
        total_chemicals - total_chemicals // 2,
    ]

    # Ensure the labels list matches the number of chemicals
    if labels is not None:
        assert (
            len(labels) == total_chemicals
        ), "Labels must match the number of chemicals."

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Define color palettes
    colors1 = plt.cm.viridis(
        np.linspace(0, 0.95, num_chemicals_subplots[0])
    )  # First 15 chemicals
    colors2 = plt.cm.viridis(
        np.linspace(0, 0.95, num_chemicals_subplots[1])
    )  # Remaining 14 chemicals

    # Plot first set of chemicals on ax1
    for chem_idx in range(num_chemicals_subplots[0]):
        color = colors1[chem_idx]
        gt = data[:, chem_idx]
        label = labels[chem_idx] if labels is not None else f"Chemical {chem_idx + 1}"
        ax1.plot(timesteps, gt, "-", color=color, label=label)

    # Plot second set of chemicals on ax2
    for chem_idx in range(num_chemicals_subplots[0], total_chemicals):
        color = colors2[chem_idx - num_chemicals_subplots[0]]
        gt = data[:, chem_idx]
        label = labels[chem_idx] if labels is not None else f"Chemical {chem_idx + 1}"
        ax2.plot(timesteps, gt, "-", color=color, label=label)

    # Set labels and title
    ax1.set_xlabel("Time")
    ax1.set_ylabel("log(Chemical Abundance)")
    ax2.set_xlabel("Time")

    # Remove individual subplot titles and add a global title
    fig.suptitle(r"Example Trajectories: $\tt{osu2008}$ Dataset", fontsize=16)
    # fig.subplots_adjust(wspace=0.05)  # Adjust space between subplots

    # Place the legend of the left plot between the two subplots
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(
        handles1,
        labels1,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
    )

    # Legend for the second plot can stay outside on the right if needed
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles2,
        labels2,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
    )

    # Add grid to both subplots
    ax1.grid(True)
    ax2.grid(True)

    # Set x-axis limits tightly
    for ax in [ax1, ax2]:
        ax.set_xlim(timesteps.min(), timesteps.max())

    # Remove tick marks but keep labels and gridlines
    ax1.tick_params(axis="both", which="both", length=0)
    ax2.tick_params(axis="both", which="both", length=0)

    # Adjust layout to make room for the legend between the subplots
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the rect parameter

    # Save the plot if required
    if save:
        # Create a dummy conf dictionary
        conf = {
            "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
            "verbose": True,
        }

        # Call save_plot with the dummy conf
        save_plot(
            plt,
            "example_trajectories_paper.png",
            conf,
            dpi=200,
            base_dir="datasets",
            increase_count=False,
        )

    plt.show()


def plot_example_trajectories_poster(
    dataset_name: str,
    data: np.ndarray,
    timesteps: np.ndarray,
    save: bool = False,
    sample_idx: int = 0,
    labels: list[str] | None = None,
) -> None:
    """
    Plot example trajectories for the dataset with two subplots, one showing 15 chemicals and another showing the remaining.

    Args:
        dataset_name (str): The name of the dataset.
        data (np.ndarray): The data to plot.
        timesteps (np.ndarray): Timesteps array.
        save (bool, optional): Whether to save the plot as a file.
        sample_idx (int, optional): Index of the sample to plot.
        labels (list, optional): List of labels for the chemicals.
    """
    # Ensure we are plotting the correct sample
    data = data[sample_idx]

    # Define the number of chemicals per subplot
    total_chemicals = data.shape[1]
    num_chemicals_subplots = [
        total_chemicals // 2,
        total_chemicals - total_chemicals // 2,
    ]

    # Ensure the labels list matches the number of chemicals
    if labels is not None:
        assert (
            len(labels) == total_chemicals
        ), "Labels must match the number of chemicals."

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4), sharey=True
    )  # Reduced horizontal width

    # Define color palettes
    colors1 = plt.cm.viridis(
        np.linspace(0, 0.95, num_chemicals_subplots[0])
    )  # First 15 chemicals
    colors2 = plt.cm.viridis(
        np.linspace(0, 0.95, num_chemicals_subplots[1])
    )  # Remaining 14 chemicals

    # Plot first set of chemicals on ax1
    for chem_idx in range(num_chemicals_subplots[0]):
        color = colors1[chem_idx]
        gt = data[:, chem_idx]
        label = labels[chem_idx] if labels is not None else f"Chemical {chem_idx + 1}"
        ax1.plot(timesteps, gt, "-", color=color, label=label)

    # Plot second set of chemicals on ax2
    for chem_idx in range(num_chemicals_subplots[0], total_chemicals):
        color = colors2[chem_idx - num_chemicals_subplots[0]]
        gt = data[:, chem_idx]
        label = labels[chem_idx] if labels is not None else f"Chemical {chem_idx + 1}"
        ax2.plot(timesteps, gt, "-", color=color, label=label)

    # Set labels and title
    ax1.set_xlabel("Time", fontsize=12)  # Increased font size
    ax1.set_ylabel("log(Chemical Abundance)", fontsize=12)  # Increased font size
    ax2.set_xlabel("Time", fontsize=12)  # Increased font size

    # Remove individual subplot titles and add a global title
    fig.suptitle(
        r"Example Trajectories: $\tt{osu2008}$ Dataset", fontsize=14
    )  # Slightly larger title
    # fig.subplots_adjust(wspace=0.05)  # Adjust space between subplots

    # Place the legend of the left plot between the two subplots
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(
        handles1,
        labels1,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
        fontsize=10,  # Increased font size for legend
    )

    # Legend for the second plot can stay outside on the right if needed
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles2,
        labels2,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
        fontsize=10,  # Increased font size for legend
    )

    # Add grid to both subplots
    ax1.grid(True)
    ax2.grid(True)

    # Set x-axis limits tightly
    for ax in [ax1, ax2]:
        ax.set_xlim(timesteps.min(), timesteps.max())

    # Remove tick marks but keep labels and gridlines
    ax1.tick_params(
        axis="both", which="both", length=0, labelsize=10
    )  # Adjust tick label size
    ax2.tick_params(
        axis="both", which="both", length=0, labelsize=10
    )  # Adjust tick label size

    # Adjust layout to make room for the legend between the subplots
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the rect parameter

    # Save the plot if required
    if save:
        # Create a dummy conf dictionary
        conf = {
            "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
            "verbose": True,
        }

        # Call save_plot with the dummy conf
        save_plot(
            plt,
            "example_trajectories_poster.png",
            conf,
            dpi=200,
            base_dir="datasets",
            increase_count=False,
        )

    plt.show()


def plot_initial_conditions_distribution(
    dataset_name: str,
    train_data: np.ndarray,
    test_data: np.ndarray,
    chemical_names: list[str] = None,
    max_chemicals: int = 10,
    log: bool = True,
    quantities_per_plot: int = 10,
) -> None:
    """
    Plot the distribution of initial conditions (t=0) for each chemical from both train and test datasets.

    Args:
        dataset_name (str): The name of the dataset (e.g., "osu2008").
        train_data (np.ndarray): Training dataset array of shape [num_samples, num_timesteps, num_chemicals].
        test_data (np.ndarray): Testing dataset array of shape [num_samples, num_timesteps, num_chemicals].
        chemical_names (list, optional): List of chemical names for labeling the lines.
        max_chemicals (int, optional): Maximum number of chemicals to plot. Default is 10.
        log (bool, optional): Whether the data is in log-space and should be exponentiated (i.e., data = 10**data).
        quantities_per_plot (int, optional): Number of quantities to plot per subplot. Default is 10.
    """
    # If data is in log space, exponentiate it to get linear-scale data
    if log:
        train_data = 10**train_data
        test_data = 10**test_data

    # Extract initial conditions (t=0) for both train and test datasets
    train_initial_conditions = train_data[
        :, 0, :
    ]  # Shape: [num_samples, num_chemicals]
    test_initial_conditions = test_data[:, 0, :]  # Shape: [num_samples, num_chemicals]

    # Cap the number of chemicals to plot at max_chemicals (default 10)
    num_chemicals = min(max_chemicals, train_initial_conditions.shape[1])
    train_initial_conditions = train_initial_conditions[:, :num_chemicals]
    test_initial_conditions = test_initial_conditions[:, :num_chemicals]

    chemical_names = (
        chemical_names[:num_chemicals]
        if chemical_names is not None
        else [f"Chemical {i+1}" for i in range(num_chemicals)]
    )

    # Split the chemicals into groups of 10 for plotting
    num_plots = int(np.ceil(num_chemicals / quantities_per_plot))

    # Initialize lists to hold processed conditions
    processed_train_conditions = []
    processed_test_conditions = []
    zero_counts_train = 0
    zero_counts_test = 0

    for i in range(num_chemicals):
        train_cond = train_initial_conditions[:, i]
        test_cond = test_initial_conditions[:, i]

        if log:
            # For log-scaled histograms, filter out non-positive values and take log10
            non_zero_train = train_cond > 0
            non_zero_test = test_cond > 0

            filtered_train = train_cond[non_zero_train]
            filtered_test = test_cond[non_zero_test]

            # Take log10 for histogram binning
            log_train = np.log10(filtered_train)
            log_test = np.log10(filtered_test)

            processed_train_conditions.append(log_train)
            processed_test_conditions.append(log_test)

            # Count zeros
            zero_counts_train += np.sum(~non_zero_train)
            zero_counts_test += np.sum(~non_zero_test)
        else:
            # For linear-scaled histograms, include all data
            processed_train_conditions.append(train_cond)
            processed_test_conditions.append(test_cond)

            zero_counts_train = 0
            zero_counts_test = 0

    # Determine global min and max for binning
    all_train = np.concatenate(
        [cond for cond in processed_train_conditions if len(cond) > 0]
    )
    all_test = np.concatenate(
        [cond for cond in processed_test_conditions if len(cond) > 0]
    )

    if len(all_train) == 0 and len(all_test) == 0:
        raise ValueError("No data available for plotting after filtering.")

    global_min = min(np.min(all_train), np.min(all_test))
    global_max = max(np.max(all_train), np.max(all_test))

    # Define bin edges based on the scaling
    num_bins = 100
    bins = np.linspace(global_min, global_max, num_bins)

    # Create subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable

    colors = plt.cm.viridis(np.linspace(0, 0.9, quantities_per_plot))

    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        start_idx = plot_idx * quantities_per_plot
        end_idx = min((plot_idx + 1) * quantities_per_plot, num_chemicals)

        for i in range(start_idx, end_idx):
            # Compute histogram for train dataset
            train_hist, train_bin_edges = np.histogram(
                processed_train_conditions[i], bins=bins, density=True
            )
            # Smooth the histogram with a Gaussian filter
            train_smoothed_hist = gaussian_filter1d(train_hist, sigma=1)

            # Compute histogram for test dataset
            test_hist, test_bin_edges = np.histogram(
                processed_test_conditions[i], bins=bins, density=True
            )
            # Smooth the histogram with a Gaussian filter
            test_smoothed_hist = gaussian_filter1d(test_hist, sigma=1)

            if log:
                # Convert bin edges back to linear scale for plotting
                x_train = 10 ** train_bin_edges[:-1]
                x_test = 10 ** test_bin_edges[:-1]
            else:
                # Use linear bin edges directly
                x_train = train_bin_edges[:-1]
                x_test = test_bin_edges[:-1]

            # Plot the smoothed histogram for the training data
            ax.plot(
                x_train,
                train_smoothed_hist,
                label=chemical_names[i],
                color=colors[i % quantities_per_plot],
            )

            # Plot the smoothed histogram for the test data with dashed line
            ax.plot(
                x_test,
                test_smoothed_hist,
                "--",
                color=colors[i % quantities_per_plot],
            )

        ax.set_ylabel("Density (PDF)")
        ax.set_ylim(
            0, 1.2 * max(np.max(train_smoothed_hist), np.max(test_smoothed_hist))
        )
        if log:
            ax.set_xscale("log")
            ax.set_xlim(10**global_min, 10**global_max)
        else:
            ax.set_xlim(global_min, global_max)
        if chemical_names is not None:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(
                    1.03,
                    0.5,
                ),
                borderaxespad=0.0,
            )

    plt.xlabel("Initial Condition Value")
    fig.suptitle(
        f"Initial Condition Distribution per Chemical (Dataset: {dataset_name}, "
        f"{train_data.shape[0]} train samples, {test_data.shape[0]} test samples)"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    conf = {
        "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
        "verbose": True,
    }

    save_plot(
        plt,
        "IC_distribution.png",
        conf,
        dpi=200,
        base_dir="datasets",
        increase_count=False,
    )

    plt.close()


def plot_average_gradients_over_time(
    dataset_name: str,
    train_data: np.ndarray,
    chemical_names: list[str] | None = None,
    max_quantities: int = 10,
) -> None:
    """
    Plot the average gradient of each quantity in the train dataset over time.

    Args:
        dataset_name (str): The name of the dataset (e.g., "osu2008").
        train_data (np.ndarray): Training dataset array of shape [n_samples, n_timesteps, n_quantities].
        chemical_names (list, optional): List of chemical names for labeling the lines.
        max_quantities (int, optional): Maximum number of quantities to plot. Default is 10.
    """
    # Cap the number of quantities to plot at 50
    num_quantities = min(max_quantities, 50, train_data.shape[2])
    train_data = train_data[:, :, :num_quantities]

    chemical_names = (
        chemical_names[:num_quantities] if chemical_names is not None else None
    )

    # Calculate the gradient for each quantity at each timestep for all samples
    gradients = np.gradient(train_data, axis=1)  # Calculate gradient along time axis

    # Average the gradients over all samples (axis 0)
    avg_gradients = np.mean(gradients, axis=0)  # Shape [n_timesteps, n_quantities]

    # Create a time vector assuming timesteps are equally spaced
    n_timesteps = train_data.shape[1]
    time = np.arange(n_timesteps)

    # Split the quantities into groups of 10
    quantities_per_plot = 10
    num_plots = int(np.ceil(num_quantities / quantities_per_plot))

    # Create subplots with shared x-axis
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even if there's only one plot

    # Get custom color palette
    colors = get_custom_palette(quantities_per_plot)

    # Plot the average gradient for each quantity over time
    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        start_idx = plot_idx * quantities_per_plot
        end_idx = min((plot_idx + 1) * quantities_per_plot, num_quantities)

        for i in range(start_idx, end_idx):
            # Plot the average gradient of the current quantity
            ax.plot(
                time,
                avg_gradients[:, i],
                label=(
                    chemical_names[i]
                    if chemical_names is not None and len(chemical_names) > i
                    else f"Quantity {i + 1}"
                ),
                color=colors[i % quantities_per_plot],
            )

        ax.set_xlabel("Time")
        ax.set_xlim(time[0], time[-1])
        ax.set_ylabel("Average Gradient")
        if chemical_names is not None:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(
                    1.03,
                    0.5,
                ),
                borderaxespad=0.0,
            )

    fig.suptitle(
        f"Average Gradient of Each Quantity Over Time (Dataset: {dataset_name})"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Saving the plot
    conf = {
        "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
        "verbose": True,
    }

    save_plot(
        plt,
        "average_gradients.png",  # Save the plot with this name
        conf,
        dpi=200,
        base_dir="datasets",  # Base directory for saving the plot
        increase_count=False,
    )

    plt.close()


def plot_all_gradients_over_time(
    dataset_name: str,
    train_data: np.ndarray,
    chemical_names: list[str] | None = None,
    max_quantities: int = 10,
    spread: float = 0.01,  # Spread for Gaussian noise
    noise_smoothing: float = 2.0,  # Controls how smooth the noise is along the trajectory
    quantities_per_plot: int = 6,
) -> None:
    """
    Plot the average gradient of each quantity in the train dataset over time,
    with individual sample trajectories shown with low opacity and smooth Gaussian spread.

    Args:
        dataset_name (str): The name of the dataset (e.g., "osu2008").
        train_data (np.ndarray): Training dataset array of shape [n_samples, n_timesteps, n_quantities].
        chemical_names (list, optional): List of chemical names for labeling the lines.
        max_quantities (int, optional): Maximum number of quantities to plot. Default is 10.
        spread (float, optional): Spread for adding Gaussian noise to the trajectories. Default is 0.05.
        noise_smoothing (float, optional): Sigma for smoothing the noise along the trajectory. Default is 2.0.
        quantities_per_plot (int, optional): Number of quantities to plot per subplot. Default is 6.
    """
    # Ensure that no more than 1000 trajectories are plotted
    if train_data.shape[0] > 1000:
        factor = (train_data.shape[0] // 1000) + 1
        train_data = train_data[::factor]
    # Cap the number of quantities to plot at 50
    num_quantities = min(max_quantities, 50, train_data.shape[2])
    train_data = train_data[:, :, :num_quantities]

    chemical_names = (
        chemical_names[:num_quantities] if chemical_names is not None else None
    )

    # Calculate the gradient for each quantity at each timestep for all samples
    gradients = np.gradient(
        train_data, axis=1
    )  # Shape [n_samples, n_timesteps, n_quantities]

    # Average the gradients over all samples (axis 0)
    avg_gradients = np.mean(gradients, axis=0)  # Shape [n_timesteps, n_quantities]

    # Create a time vector assuming timesteps are equally spaced
    n_timesteps = train_data.shape[1]
    time = np.arange(n_timesteps)

    # Split the quantities into groups
    num_plots = int(np.ceil(num_quantities / quantities_per_plot))

    # Create subplots with shared x-axis
    fig, axes = plt.subplots(num_plots, 1, figsize=(9, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even if there's only one plot

    # Get custom color palette
    colors = plt.cm.viridis(np.linspace(0, 0.9, quantities_per_plot))

    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        start_idx = plot_idx * quantities_per_plot
        end_idx = min((plot_idx + 1) * quantities_per_plot, num_quantities)

        for i in range(start_idx, end_idx):
            # Plot all individual trajectories with low opacity and some smooth Gaussian spread
            for sample_idx in range(train_data.shape[0]):
                # Generate smooth Gaussian noise across the trajectory
                noise = np.random.normal(0, spread, n_timesteps)
                smooth_noise = gaussian_filter1d(noise, sigma=noise_smoothing)

                # Add smooth noise to the gradient and plot
                noisy_gradients = gradients[sample_idx, :, i] + smooth_noise
                ax.plot(
                    time,
                    noisy_gradients,
                    color=colors[i % quantities_per_plot],
                    alpha=0.01,  # Very low opacity for each individual trajectory
                )

            # Plot the average gradient of the current quantity
            ax.plot(
                time,
                avg_gradients[:, i],
                label=(
                    chemical_names[i]
                    if chemical_names is not None and len(chemical_names) > i
                    else f"Quantity {i + 1}"
                ),
                color=colors[i % quantities_per_plot],
                linewidth=1,  # Make the average line more visible
            )

        ax.set_xlabel("Time")
        ax.set_xlim(time[0], time[-1])
        ax.set_ylabel("Gradient")
        if chemical_names is not None:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(
                    1.03,
                    0.5,
                ),
                borderaxespad=0.0,
            )

    fig.suptitle(
        f"Overview over Gradients for each Quantity over Time (Dataset: {dataset_name})"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Saving the plot
    conf = {
        "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
        "verbose": True,
    }

    save_plot(
        plt,
        "all_gradients.png",  # Save the plot with this name
        conf,
        dpi=200,
        base_dir="datasets",  # Base directory for saving the plot
        increase_count=False,
    )

    plt.close()


def plot_all_trajectories_over_time(
    dataset_name: str,
    train_data: np.ndarray,
    chemical_names: list[str] | None = None,
    max_quantities: int = 10,
    spread: float = 0.01,  # Spread for Gaussian noise
    quantities_per_plot: int = 6,
    log: bool = False,
) -> None:
    """
    Plot the average gradient of each quantity in the train dataset over time,
    with individual sample trajectories shown with low opacity and some Gaussian spread.

    Args:
        dataset_name (str): The name of the dataset (e.g., "osu2008").
        train_data (np.ndarray): Training dataset array of shape [n_samples, n_timesteps, n_quantities].
        chemical_names (list, optional): List of chemical names for labeling the lines.
        max_quantities (int, optional): Maximum number of quantities to plot. Default is 10.
        spread (float, optional): Spread for adding Gaussian noise to the trajectories. Default is 0.05.
        quantities_per_plot (int, optional): Number of quantities to plot per subplot. Default is 6.
        log (bool, optional): Whether the data is in log-space (only for axis labels). Default is False.
    """
    # Ensure that no more than 1000 trajectories are plotted
    if train_data.shape[0] > 1000:
        factor = (train_data.shape[0] // 1000) + 1
        train_data = train_data[::factor]
    # Cap the number of quantities to plot at 50
    num_quantities = min(max_quantities, 50, train_data.shape[2])
    train_data = train_data[:, :, :num_quantities]

    chemical_names = (
        chemical_names[:num_quantities] if chemical_names is not None else None
    )

    # Average the gradients over all samples (axis 0)
    avg_trajectories = np.mean(train_data, axis=0)  # Shape [n_timesteps, n_quantities]

    # Create a time vector assuming timesteps are equally spaced
    n_timesteps = train_data.shape[1]
    time = np.arange(n_timesteps)

    # Split the quantities into groups of quantities_per_plot
    num_plots = int(np.ceil(num_quantities / quantities_per_plot))

    # Create subplots with shared x-axis
    fig, axes = plt.subplots(num_plots, 1, figsize=(9, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even if there's only one plot

    # Get custom color palette
    # colors = get_custom_palette(quantities_per_plot)
    colors = plt.cm.viridis(np.linspace(0, 0.9, quantities_per_plot))

    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        start_idx = plot_idx * quantities_per_plot
        end_idx = min((plot_idx + 1) * quantities_per_plot, num_quantities)

        for i in range(start_idx, end_idx):
            # Plot all individual trajectories with low opacity and some Gaussian spread
            for sample_idx in range(train_data.shape[0]):
                noisy_gradients = train_data[sample_idx, :, i] + np.random.normal(
                    0, spread, n_timesteps
                )
                ax.plot(
                    time,
                    noisy_gradients,
                    color=colors[i % quantities_per_plot],
                    alpha=0.01,  # Very low opacity for each individual trajectory
                )

            # Plot the average gradient of the current quantity
            ax.plot(
                time,
                avg_trajectories[:, i],
                label=(
                    chemical_names[i]
                    if chemical_names is not None and len(chemical_names) > i
                    else f"Quantity {i + 1}"
                ),
                color=colors[i % quantities_per_plot],
                linewidth=1,  # Make the average line more visible
            )

        ax.set_xlabel("Time")
        ax.set_xlim(time[0], time[-1])
        ylabel = "log(Abundance)" if log else "Abundance"
        ax.set_ylabel(ylabel)
        if chemical_names is not None:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(
                    1.03,
                    0.5,
                ),
                borderaxespad=0.0,  # Remove padding
            )

    fig.suptitle(
        f"Overview over Trajectories for each Quantity over Time (Dataset: {dataset_name})"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Saving the plot
    conf = {
        "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
        "verbose": True,
    }

    save_plot(
        plt,
        "all_trajectories.png",  # Save the plot with this name
        conf,
        dpi=200,
        base_dir="datasets",  # Base directory for saving the plot
        increase_count=False,
    )

    plt.close()


def debug_numerical_errors_plot(
    dataset_name: str,
    train_data: np.ndarray,
    chemical_names: list[str] | None = None,
    max_quantities: int = 10,
    threshold: float = 0.1,
    max_faulty: int = 10,
    quantities_per_plot: int = 10,
) -> None:
    """
    Plot faulty trajectories with gradients exceeding a threshold.

    Args:
        dataset_name (str): The name of the dataset.
        train_data (np.ndarray): Dataset array of shape [n_samples, n_timesteps, n_quantities].
        chemical_names (list, optional): Names for each quantity.
        max_quantities (int, optional): Maximum number of quantities to plot. Default is 10.
        threshold (float, optional): Gradient threshold to define faulty trajectories. Default is 0.1.
        max_faulty (int, optional): Maximum number of faulty trajectories to plot. Default is 10.
        quantities_per_plot (int, optional): Number of quantities to plot per subplot. Only for faulty IC plot.
    """
    # Ensure max_faulty does not exceed 10
    max_faulty = min(max_faulty, 10)

    n_samples, n_timesteps, n_quantities = train_data.shape

    # Cap the number of quantities to plot
    num_quantities = min(max_quantities, n_quantities)
    train_data = train_data[:, :, :num_quantities]

    if chemical_names is not None:
        chemical_names = chemical_names[:num_quantities]
    else:
        chemical_names = [f"Quantity {i+1}" for i in range(num_quantities)]

    # Identify faulty trajectories
    gradients = np.gradient(train_data, axis=1)
    faulty_mask = np.any(np.abs(gradients) > threshold, axis=(1, 2))
    faulty_indices = np.where(faulty_mask)[0]

    if faulty_indices.size == 0:
        print("No faulty trajectories found.")
        return

    plot_faulty_initial_conditions_distribution(
        dataset_name=dataset_name,
        train_data=train_data,
        faulty_indices=faulty_indices,
        chemical_names=chemical_names,
        max_chemicals=max_quantities,
        log=True,  # Set to False if data is not in log-space
        quantities_per_plot=quantities_per_plot,
    )

    # Limit the number of faulty trajectories to plot
    faulty_indices = faulty_indices[:max_faulty]
    time = np.linspace(0, 1, n_timesteps)

    # Plot each faulty trajectory
    fig, axes = plt.subplots(
        len(faulty_indices), 1, figsize=(9, 4 * len(faulty_indices))
    )

    if len(faulty_indices) == 1:
        axes = [axes]

    for idx, sample_idx in enumerate(faulty_indices):
        ax = axes[idx]
        trajectory = train_data[sample_idx]
        traj_gradients = gradients[sample_idx]

        faulty_quantities = np.any(np.abs(traj_gradients) > threshold, axis=0)
        faulty_labels = [
            chemical_names[q]
            for q, is_faulty in enumerate(faulty_quantities)
            if is_faulty
        ]

        for q in range(num_quantities):
            ax.plot(time, trajectory[:, q], label=chemical_names[q], linewidth=1.5)
            if faulty_quantities[q]:
                spike_timesteps = np.where(np.abs(traj_gradients[:, q]) > threshold)[0]
                ax.scatter(
                    time[spike_timesteps],
                    trajectory[spike_timesteps, q],
                    color="red",
                    s=20,
                )

        ax.set_title(
            f"Sample {sample_idx}, Faulty Quantities: {', '.join(faulty_labels)}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Chemical Abundance")
        ax.legend(loc="upper right")

    plt.tight_layout()
    fig.suptitle(
        f"Faulty Trajectories (Dataset: {dataset_name}, Max: {max_faulty})", fontsize=16
    )

    # Saving the plot
    conf = {
        "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
        "verbose": True,
    }

    save_plot(
        plt,
        "faulty_trajectories.png",  # Save the plot with this name
        conf,
        dpi=200,
        base_dir="datasets",  # Base directory for saving the plot
        increase_count=False,
    )


def plot_faulty_initial_conditions_distribution(
    dataset_name: str,
    train_data: np.ndarray,
    faulty_indices: np.ndarray,
    chemical_names: list[str] | None = None,
    max_chemicals: int = 10,
    log: bool = True,
    quantities_per_plot: int = 10,
) -> None:
    """
    Plot the distribution of initial conditions (t=0) for each chemical from faulty trajectories.

    Args:
        dataset_name (str): The name of the dataset (e.g., "osu2008").
        train_data (np.ndarray): Training dataset array of shape [num_samples, num_timesteps, num_chemicals].
        faulty_indices (np.ndarray): Indices of faulty trajectories within train_data.
        chemical_names (list, optional): List of chemical names for labeling the lines.
        max_chemicals (int, optional): Maximum number of chemicals to plot. Default is 10.
        log (bool, optional): Whether the data is in log-space and should be exponentiated (i.e., data = 10**data).
        quantities_per_plot (int, optional): Number of quantities to plot per subplot. Default is 10.
    """
    # If data is in log space, exponentiate it
    if log:
        train_data = 10**train_data

    # Extract initial conditions (t=0) for faulty trajectories
    faulty_data = train_data[faulty_indices, :, :]
    faulty_initial_conditions = faulty_data[:, 0, :]  # Extract t=0

    # Cap the number of chemicals to plot at max_chemicals
    num_chemicals = min(max_chemicals, faulty_initial_conditions.shape[1])
    faulty_initial_conditions = faulty_initial_conditions[:, :num_chemicals]

    if chemical_names is not None:
        chemical_names = chemical_names[:num_chemicals]
    else:
        chemical_names = [f"Chemical {i+1}" for i in range(num_chemicals)]

    # Split the chemicals into groups of 10 for plotting
    num_plots = int(np.ceil(num_chemicals / quantities_per_plot))

    # Initialize list to hold log-transformed non-zero initial conditions
    log_faulty_conditions = []
    zero_counts_faulty = 0

    # Transform data to log-space and filter out zeros
    for i in range(num_chemicals):
        faulty_chemical_conditions = faulty_initial_conditions[:, i]

        non_zero_faulty_conditions = faulty_chemical_conditions[
            faulty_chemical_conditions > 0
        ]

        if log:
            log_faulty_conditions.append(np.log10(non_zero_faulty_conditions))
        else:
            log_faulty_conditions.append(non_zero_faulty_conditions)

        zero_counts_faulty += np.sum(faulty_chemical_conditions == 0)

    # Determine global min and max for histogram bins
    min_values = [np.min(cond) for cond in log_faulty_conditions if len(cond) > 0]
    max_values = [np.max(cond) for cond in log_faulty_conditions if len(cond) > 0]

    if not min_values or not max_values:
        print("No non-zero initial conditions to plot.")
        return

    global_min = np.min(min_values)
    global_max = np.max(max_values)

    # Set up the x-axis range to nearest whole numbers in log-space
    x_min = np.floor(global_min)
    x_max = np.ceil(global_max)

    # Define the x-axis range for plotting
    x_vals = np.linspace(x_min, x_max + 0.1, 100)

    # Create subplots with shared x-axis
    fig, axes = plt.subplots(num_plots, 1, figsize=(9, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even if there's only one plot

    # Get custom color palette
    # Assuming get_custom_palette is defined elsewhere
    colors = plt.cm.viridis(np.linspace(0, 0.9, quantities_per_plot))

    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        start_idx = plot_idx * quantities_per_plot
        end_idx = min((plot_idx + 1) * quantities_per_plot, num_chemicals)

        for i in range(start_idx, end_idx):
            # Compute histogram for faulty dataset
            faulty_hist, faulty_bin_edges = np.histogram(
                log_faulty_conditions[i], bins=x_vals, density=True
            )
            # Smooth the histogram with a Gaussian filter
            faulty_smoothed_hist = gaussian_filter1d(faulty_hist, sigma=1)

            # Plot the smoothed histogram for the faulty data
            ax.plot(
                10 ** faulty_bin_edges[:-1],  # Convert back to original scale
                faulty_smoothed_hist,
                label=chemical_names[i] if chemical_names is not None else None,
                color=colors[i % quantities_per_plot],
            )

        ax.set_yscale("linear")
        ax.set_ylabel("Density (PDF)")
        ax.set_xlim(10**x_min, 10**x_max)
        if chemical_names is not None:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(
                    1.03,
                    0.5,
                ),
                borderaxespad=0.0,
            )

    plt.xscale("log")  # Log scale for initial conditions magnitudes
    plt.xlabel("Initial Condition Magnitude")
    fig.suptitle(
        f"Initial Condition Distribution per Chemical for Faulty Trajectories (Dataset: {dataset_name}, {len(faulty_indices)} faulty samples)"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot
    conf = {
        "training_id": dataset_name.lower(),  # Use dataset_name as the training_id
        "verbose": True,
    }

    save_plot(
        plt,
        "Faulty_IC_distribution.png",
        conf,
        dpi=200,
        base_dir="datasets",
        increase_count=False,
    )

    plt.close()
