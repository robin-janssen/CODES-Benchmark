import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(1, "../..")

from codes import save_plot


def plot_example_trajectories(
    dataset_name: str,
    data: np.ndarray,
    timesteps: np.ndarray,
    num_chemicals: int = 10,
    labels: list[str] | None = None,
    save: bool = False,
    sample_idx: int = 0,
    log: bool = False,
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
    """
    # Cap the number of chemicals at the number of available chemicals
    num_chemicals = min(data.shape[2], num_chemicals)
    data = data[sample_idx]

    # Define the color palette
    colors = plt.cm.viridis(np.linspace(0, 1, num_chemicals))

    # Create a single plot
    plt.figure(figsize=(12, 6))

    for chem_idx in range(num_chemicals):
        color = colors[chem_idx]
        gt = data[:, chem_idx]

        # Plot ground truth
        plt.plot(
            timesteps,
            gt,
            "-",
            color=color,
            label=f"Quantity {chem_idx + 1}" if labels is None else labels[chem_idx],
        )

    # Set labels and title
    plt.xlim = [timesteps[0], timesteps[-1]]
    plt.xlabel("Time")
    # Remove all ticks
    plt.tick_params(axis="x", which="both", bottom=False, top=False)
    plt.tick_params(axis="y", which="both", left=False, right=False)
    ylabel = "log(Abundance)" if log else "Abundance"
    plt.ylabel(ylabel)
    plt.title(f"Example Trajectories for Dataset: {dataset_name}")
    plt.legend(title="Quantity")
    plt.grid(True)

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
            "example_trajectories.png",
            conf,
            dpi=300,
            base_dir="datasets",
            increase_count=False,
        )

    # plt.show()


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
            dpi=300,
            base_dir="datasets",
            increase_count=False,
        )

    plt.show()
