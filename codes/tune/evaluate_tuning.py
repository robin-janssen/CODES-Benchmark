import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_loss_history(model_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load loss histories from a saved model file.

    The saved file is expected to be in the custom format, where the loss histories and other
    attributes are stored under the "attributes" key.

    Args:
        model_path (str): Path to the .pth file.

    Returns:
        tuple: (train_loss, test_loss, n_epochs)
    """
    model_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    attributes = model_dict.get("attributes", {})
    # Expect that train_loss, test_loss, and n_epochs have been saved.
    train_loss = (
        np.array(attributes.get("train_loss"))
        if attributes.get("train_loss") is not None
        else None
    )
    test_loss = (
        np.array(attributes.get("test_loss"))
        if attributes.get("test_loss") is not None
        else None
    )
    n_epochs = attributes.get(
        "n_epochs", len(train_loss) if train_loss is not None else 0
    )
    return train_loss, test_loss, n_epochs


def plot_losses(
    loss_histories: tuple[np.ndarray, ...],
    epochs: int,
    labels: tuple[str, ...],
    title: str = "Losses",
    save: bool = False,
    conf: dict | None = None,
    surr_name: str | None = None,
    mode: str = "main",
    percentage: float = 2.0,
    show_title: bool = True,
) -> None:
    """
    Plot the loss trajectories for multiple models using their actual lengths.

    Each loss trajectory is plotted over its own length (i.e. trial-specific number of epochs),
    rather than forcing all trajectories to the length of the shortest one. The global y-axis limits
    are determined from the valid (nonzero) portions of each trajectory after excluding the initial
    percentage of epochs.

    Args:
        loss_histories (tuple[np.ndarray, ...]): Tuple of loss history arrays.
        epochs (int): Total number of training epochs (used for labeling only).
        labels (tuple[str, ...]): Labels for each loss history.
        title (str): Title for the plot.
        save (bool): Whether to save the plot as an image file.
        conf (dict | None): Configuration dictionary (used for naming output files).
        surr_name (str | None): Surrogate model name.
        mode (str): Mode for labeling (e.g., "main" or surrogate name).
        percentage (float): Percentage of initial epochs to exclude from min/max y-value calculation.
        show_title (bool): Whether to display the title.
    """
    # Filter out loss arrays that are None or empty.
    valid_losses = [
        loss for loss in loss_histories if loss is not None and loss.size > 0
    ]
    if not valid_losses:
        print("No valid loss arrays found; skipping plot.")
        return

    # Determine global maximum length (for x-axis limit).
    lengths = [len(loss) for loss in valid_losses]
    max_length = max(lengths)

    # Compute global min and max values across all valid losses.
    valid_mins = []
    valid_maxes = []
    for loss in valid_losses:
        start_idx = int(len(loss) * (percentage / 100))
        slice_vals = loss[start_idx:]
        valid_vals = slice_vals[slice_vals > 0]
        if valid_vals.size > 0:
            valid_mins.append(valid_vals.min())
            valid_maxes.append(valid_vals.max())
    if valid_mins:
        global_min = min(valid_mins)
        global_max = max(valid_maxes)
    else:
        global_min, global_max = 1e-8, 1.0

    # Create color map for plotting.
    colors = plt.cm.magma(np.linspace(0.15, 0.85, len(loss_histories)))

    plt.figure(figsize=(6, 4))
    loss_plotted = False
    for loss, label in zip(loss_histories, labels):
        if loss is not None and loss.size > 0:
            # Generate x-axis based on the actual length of this loss history.
            x_epochs = np.arange(len(loss))
            plt.plot(x_epochs, loss, label=label, color=colors[labels.index(label)])
            loss_plotted = True

    plt.xlabel("Epoch")
    plt.xlim(0, max_length)
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.ylim(global_min, global_max)
    if show_title:
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

    # Save the plot if requested.
    if save and conf and surr_name:
        out_dir = os.path.join("tuned", conf.get("study_name", "study"))
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"losses_{mode.lower()}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.close()


def evaluate_tuning(study_name: str) -> None:
    """
    Evaluate the tuning step by generating loss plots for each surrogate model.

    This function looks for folders in "tuned/<study_name>/models". Each folder should
    correspond to a surrogate model (e.g., "FullyConnected", "LatentPoly", etc.). It then
    loads all .pth files within each folder, extracts the loss trajectories (test_loss),
    extracts the trial number from the filename, and generates a loss plot.

    Args:
        study_name (str): Name of the study (e.g., "primordialtest").
    """
    models_dir = os.path.join("tuned", study_name, "models")
    output_dir = os.path.join("tuned", study_name)
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of surrogate folders.
    surrogate_folders = [
        d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))
    ]
    for surr_folder in surrogate_folders:
        surr_path = os.path.join(models_dir, surr_folder)
        print(f"Processing surrogate model folder: {surr_folder}")

        # Find all model files (*.pth) in this folder.
        model_files = [f for f in os.listdir(surr_path) if f.endswith(".pth")]
        if not model_files:
            print(f"No model files found in {surr_path}. Skipping.")
            continue

        trial_numbers = []
        test_loss_histories = []
        n_epochs = None

        for file_name in model_files:
            # Extract trial number from filename (e.g., "latentpoly_0.pth")
            match = re.search(r"_(\d+)\.pth$", file_name)
            if match:
                trial_num = int(match.group(1))
            else:
                trial_num = -1  # Default if extraction fails.
            trial_numbers.append(trial_num)

            file_path = os.path.join(surr_path, file_name)
            _, test_loss, epochs = load_loss_history(file_path)
            test_loss_histories.append(test_loss)
            if n_epochs is None:
                n_epochs = epochs

        # Sort trials by trial number for consistent labeling.
        sorted_trials = sorted(
            zip(trial_numbers, test_loss_histories), key=lambda x: x[0]
        )
        trial_numbers, test_loss_histories = zip(*sorted_trials)
        labels = tuple(f"Trial {num}" for num in trial_numbers)

        # Create the plot using the provided plot_losses function.
        plot_losses(
            loss_histories=test_loss_histories,
            epochs=n_epochs,
            labels=labels,
            title=f"{surr_folder} Test Losses",
            save=True,
            conf={"study_name": study_name},
            surr_name=surr_folder,
            mode=surr_folder,
            show_title=True,
        )
        print(f"Loss plot created for surrogate: {surr_folder}.")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing study_name.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate tuning loss trajectories and generate plots."
    )
    parser.add_argument(
        "--study_name",
        type=str,
        required=True,
        help="Name of the study (e.g., primordialtest)",
    )
    return parser.parse_args()


def main():
    """
    Main function to evaluate tuning.

    Reads the study name from command-line arguments, processes each surrogate folder in
    tuned/<study_name>/models, and generates loss plots saved to tuned/<study_name>/.
    """
    args = parse_args()
    study_name = args.study_name
    evaluate_tuning(study_name)


if __name__ == "__main__":
    main()
