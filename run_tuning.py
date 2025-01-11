import argparse
import os
import queue
import time

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from tqdm import tqdm

from codes.tune import create_objective, load_yaml_config
from codes.utils import nice_print


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run multi-architecture Optuna tuning (subsequent studies)."
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="osu2008",
        help="Main study identifier. Separate sub-studies will be created for each architecture.",
    )
    return parser.parse_args()


def update_trial_progress_bar(study, trial, pbar: tqdm):
    # Called every time a trial finishes or is pruned/failed
    if trial.state in (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED):
        pbar.update(1)


def run_single_study(config, study_name):
    sname, arch = study_name.split("_")
    db_path = os.path.join(f"optuna_runs/{sname}", f"{arch}.db")
    db_url = f"sqlite:///{db_path}"
    if not config.get("optuna_logging", False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    if config["prune"]:
        epochs = config["epochs"]
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=epochs // 8, max_resource=epochs, reduction_factor=2
        )
    else:
        pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=db_url,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Build device queue for this study
    device_queue = queue.Queue()
    for dev in config["devices"]:
        device_queue.put(dev)

    objective_fn = create_objective(config, study_name, device_queue)
    n_trials = config["n_trials"]
    trial_durations = []

    def trial_complete_callback(study_, trial_):
        trial_pbar.update(1)
        # Use Optuna's datetime_start to calculate trial duration
        if trial_.datetime_start:
            duration = time.time() - trial_.datetime_start.timestamp()
            trial_durations.append(duration)
            avg_duration = sum(trial_durations) / len(trial_durations)
            remaining = n_trials - len(trial_durations)
            eta_seconds = avg_duration * remaining
            trial_pbar.set_postfix(
                {
                    "ETA": f"{eta_seconds/60:.1f}m",
                    "LastTrial": f"{duration:.1f}s",
                }
            )

    with tqdm(
        total=n_trials, desc=f"Tuning {study_name}", position=1, leave=True
    ) as trial_pbar:
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            n_jobs=len(config["devices"]),
            callbacks=[
                MaxTrialsCallback(
                    n_trials, states=[TrialState.COMPLETE, TrialState.PRUNED]
                ),
                trial_complete_callback,
            ],
        )


def run_all_studies(config, main_study_name):
    surrogates = config["surrogates"]
    total_sub_studies = len(surrogates)

    # Overall progress bar for sub-studies (e.g., "1/4 -> MultiONet, 2/4 -> LatentPoly", etc.)
    with tqdm(
        total=total_sub_studies, desc="Overall Surrogates", position=0, leave=True
    ) as arch_pbar:
        for i, surr in enumerate(surrogates, start=1):
            arch_name = surr["name"]
            study_name = f"{main_study_name}_{arch_name.lower()}"

            # Print or set a simple postfix so the user sees which study is active
            arch_pbar.set_postfix({"study": study_name})

            sub_config = {
                "batch_size": surr["batch_size"],
                "dataset": config["dataset"],
                "devices": config["devices"],  # same device list for each study
                "epochs": surr["epochs"],
                "n_trials": config["n_trials"],  # taken from top-level
                "seed": config["seed"],
                "surrogate": {"name": arch_name},
                "optuna_params": surr["optuna_params"],
                "prune": config.get("prune", True),
            }

            # Run one study fully
            run_single_study(sub_config, study_name)

            arch_pbar.update(1)  # increment sub-study progress
            # Optionally inform about completion of this sub-study
            arch_pbar.set_postfix({"done": study_name})


if __name__ == "__main__":
    nice_print("Starting Optuna tuning")
    args = parse_arguments()
    config_path = os.path.join("optuna_runs", args.study_name, "optuna_config.yaml")
    config = load_yaml_config(config_path)
    optuna_logging = config.get("optuna_logging", False)
    if not optuna_logging:
        print("Optuna logging disabled. No intermediate results will be printed.")

    # If surrogates exist, run them one by one. Else, do a single study.
    if "surrogates" in config:
        run_all_studies(config, args.study_name)
    else:
        run_single_study(config, args.study_name)

    nice_print("Optuna tuning completed!")
