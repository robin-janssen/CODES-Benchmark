import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import yaml
from optuna.visualization import (
    plot_contour,
    plot_parallel_coordinate,
    plot_param_importances,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate an Optuna study and its top models."
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="branca_latentpoly",
        help="The name of the study to evaluate.",
    )
    parser.add_argument(
        "--top_n", type=int, default=20, help="Number of top models to evaluate."
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
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_test_losses(
    test_losses: list[np.ndarray],
    labels: list[str],
    study_name: str,
    window_size: int = 5,
) -> None:
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(test_losses)))

    for i, test_loss in enumerate(test_losses):
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

    # Determine the path to the study database
    db_path = f"sqlite:///optuna_runs/studies/{study_name}.db"

    # Load the study
    study = optuna.load_study(study_name=study_name, storage=db_path)

    # Print the parameters of the optimal run
    print(f"Best trial parameters for study '{study_name}':")
    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Print the total number of runs
    print(f"\nTotal number of runs: {len(study.trials)}")

    # Ensure the plots directory exists
    plots_dir = f"optuna_runs/plots/{study_name}"
    os.makedirs(plots_dir, exist_ok=True)

    for trial in study.trials:
        trial.value = np.log10(trial.value) if trial.value is not None else None

    # Generate and save plots as HTML files
    plot_contour(study).write_html(f"{plots_dir}/contour_plot.html")
    plot_param_importances(study).write_html(f"{plots_dir}/param_importances_plot.html")
    plot_parallel_coordinate(study).write_html(
        f"{plots_dir}/parallel_coordinate_plot.html",
    )

    # Evaluate top models
    config = load_config(study_name)
    dataset_name = config["dataset"]["name"]
    surrogate_name = config["surrogate"]["name"]

    # select all trials where values are not None
    clean_trials = [trial for trial in study.trials if trial.value is not None]
    best_trials = sorted(clean_trials, key=lambda t: t.value)[:top_n]

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
