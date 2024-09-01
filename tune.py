import argparse

import optuna

from optuna_runs import create_objective, load_config_from_pyfile


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="osu_latentneuralode.py",
        help="The name of the Python config file (e.g., 'config_1.py').",
    )
    return parser.parse_args()


def run(config, study_name):
    # Save the configuration before starting the study
    # save_optuna_config(config)
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///optuna_runs/studies/{study_name}.db",
        sampler=sampler,
        load_if_exists=True,
    )

    # Create the objective function with the config
    optuna_objective = create_objective(config)
    study.optimize(optuna_objective, n_trials=config["n_trials"])


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Load the configuration from the specified Python file
    config = load_config_from_pyfile(args.config)

    study_name = args.config.split(".")[0]

    # Run the study with the loaded configuration
    run(config, study_name)
