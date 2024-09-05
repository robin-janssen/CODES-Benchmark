import argparse
import os

import optuna
from optuna.visualization import (
    plot_contour,
    plot_parallel_coordinate,
    plot_param_importances,
)


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate an Optuna study.")
    parser.add_argument(
        "--study_name",
        type=str,
        default="osu_latentneuralode",
        help="The name of the study to evaluate (e.g., osu_multionet).",
    )
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()
    study_name = args.study_name

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

    # Generate and save plots as HTML files
    plot_contour(study).write_html(f"{plots_dir}/contour_plot.html")
    plot_param_importances(study).write_html(f"{plots_dir}/param_importances_plot.html")
    plot_parallel_coordinate(study).write_html(
        f"{plots_dir}/parallel_coordinate_plot.html"
    )
    # plot_intermediate_values(study).write_html(
    #     f"{plots_dir}/intermediate_values_plot.html"
    # )


if __name__ == "__main__":
    main()
