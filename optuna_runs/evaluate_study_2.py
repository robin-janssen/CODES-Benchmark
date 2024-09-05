import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate top models from an Optuna study."
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="osu_latentneuralode",
        help="The name of the study to evaluate.",
    )
    parser.add_argument(
        "--top_n", type=int, default=10, help="Number of top models to evaluate."
    )
    return parser.parse_args()


def load_config(study_name):
    config_path = f"optuna_runs/studies/{study_name}_config.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_model_test_losses(model_path):
    model_dict = torch.load(model_path, map_location=torch.device("cpu"))
    return model_dict["attributes"]["test_loss"]


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute the moving average of the given data.

    Args:
        data (np.ndarray): Array of loss values.
        window_size (int): Size of the moving window.

    Returns:
        np.ndarray: Array of the moving averages.
    """
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_test_losses(
    test_losses: list[np.ndarray],
    labels: list[str],
    study_name: str,
    window_size: int = 5,
) -> None:
    """
    Plot the training losses for different models and their moving averages.

    Args:
        test_losses (list[np.ndarray]): List of test loss arrays for each model.
        labels (list[str]): List of labels for each model.
        study_name (str): Name of the study.
        window_size (int): Size of the moving window for averaging (default is 5).

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(test_losses)))

    for i, test_loss in enumerate(test_losses):
        # Plot the original test loss
        # plt.plot(
        #     test_loss, label=f"Test Loss: {labels[i]}", linestyle="--", color=colors[i]
        # )

        # Compute and plot the moving average
        avg_loss = moving_average(test_loss, window_size)
        plt.plot(
            np.arange(len(avg_loss)) + window_size - 1,
            avg_loss,
            label=f"Test Loss (Avg {window_size}): {labels[i]}",
            color=colors[i],
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(f"Comparison of Test Losses for {study_name}")
    plt.legend()
    plt.grid(True)

    os.makedirs(f"optuna_runs/plots/{study_name}", exist_ok=True)
    plt.savefig(f"optuna_runs/plots/{study_name}/test_losses_comparison.png")
    plt.close()


def main():
    args = parse_arguments()
    study_name = args.study_name
    top_n = args.top_n

    config = load_config(study_name)
    dataset_name = config["dataset"]["name"]
    surrogate_name = config["surrogate"]["name"]
    db_path = f"sqlite:///optuna_runs/studies/{study_name}.db"
    study = optuna.load_study(study_name=study_name, storage=db_path)
    # select all trials where values is not none
    clean_trials = [trial for trial in study.trials if trial.value is not None]
    best_trials = sorted(clean_trials, key=lambda t: t.value)[:top_n]
    # best_trials = sorted(study.trials[:115], key=lambda t: t.value)[:top_n]

    test_losses = []
    labels = []
    for trial in best_trials:
        model_path = f"optuna_runs/models/{dataset_name}/{surrogate_name}/{surrogate_name.lower()}_{trial.number}.pth"
        test_loss = load_model_test_losses(model_path)
        test_losses.append(test_loss)
        labels.append(f"Trial {trial.number} LR: {trial.params['learning_rate']:.2e}")

    plot_test_losses(test_losses, labels, study_name, window_size=1)


if __name__ == "__main__":
    main()
