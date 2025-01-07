# tune.py
import argparse
import os
import queue

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

# Import from our local "optuna_fcts.py"
from codes.tune import create_objective, load_yaml_config


def parse_arguments():
    """
    Parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization."
    )
    # We only need the study_name -> it will contain optuna_config.yaml
    parser.add_argument(
        "--study_name",
        type=str,
        default="test_study",
        help="Name of the study folder in optuna_runs/studies/ to run.",
    )
    return parser.parse_args()


def run_study(config, study_name):
    """
    Runs the Optuna study with a dynamic device queue (one device per trial).
    """

    # Make a dedicated folder for this study if not already existing
    study_folder = os.path.join("optuna_runs", "studies", study_name)
    os.makedirs(study_folder, exist_ok=True)

    # The study DB goes into this folder
    db_path = os.path.join(study_folder, f"{study_name}.db")
    db_url = f"sqlite:///{db_path}"

    # Create or load the study
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=db_url,
        sampler=sampler,
        load_if_exists=True,
    )

    # Create a queue of devices
    device_queue = queue.Queue()
    for dev in config["devices"]:
        device_queue.put(dev)

    # Build the objective function
    optuna_objective = create_objective(config, study_name, device_queue)

    # Use MaxTrialsCallback to ensure exactly `n_trials` completed
    # (even if multiple are running in parallel).
    study.optimize(
        optuna_objective,
        n_trials=config["n_trials"],
        n_jobs=len(config["devices"]),
        callbacks=[
            MaxTrialsCallback(
                config["n_trials"],
                states=[TrialState.COMPLETE, TrialState.PRUNED],
            )
        ],
    )


if __name__ == "__main__":
    args = parse_arguments()

    # Load the YAML file from: optuna_runs/studies/<study_name>/optuna_config.yaml
    config_path = os.path.join(
        "optuna_runs", "studies", args.study_name, "optuna_config.yaml"
    )
    config = load_yaml_config(config_path)

    # Run the study
    run_study(config, args.study_name)
