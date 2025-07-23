import argparse
import queue
import sys
import time
from pathlib import Path

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from tqdm import tqdm

from codes.tune import (
    create_objective,
    delete_studies_if_requested,
    initialize_optuna_database,
    load_yaml_config,
    maybe_set_runtime_threshold,
    prepare_workspace,
)
from codes.utils import download_data, nice_print


def run_single_study(config: dict, study_name: str, db_url: str):
    if not config.get("optuna_logging", False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if config["multi_objective"]:
        sampler = optuna.samplers.NSGAIISampler(
            seed=config["seed"], population_size=config["population_size"]
        )
        pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(
            study_name=study_name,
            directions=["minimize", "minimize"],
            storage=db_url,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
    else:
        sampler = optuna.samplers.TPESampler(seed=config["seed"])
        pruner = (
            optuna.pruners.HyperbandPruner(
                min_resource=config["epochs"] // 8,
                max_resource=config["epochs"],
                reduction_factor=2,
            )
            if config.get("prune", False)
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=db_url,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    device_queue = queue.Queue()
    for slot_id, dev in enumerate(config["devices"]):
        device_queue.put((dev, slot_id))

    objective_fn = create_objective(config, study_name, device_queue)
    n_trials = config["n_trials"]
    n_jobs = len(config["devices"])
    warmup_target = max(5, int(n_trials * 0.10))

    all_durations: list[float] = []

    def trial_complete_callback(study_: optuna.Study, trial_: optuna.trial.FrozenTrial):
        # progress bar update
        if trial_.state in (TrialState.COMPLETE, TrialState.PRUNED):
            trial_pbar.update(1)

        # duration/eta
        if trial_.datetime_start:
            dur = time.time() - trial_.datetime_start.timestamp()
            all_durations.append(dur)
            avg = sum(all_durations) / len(all_durations)
            eta = (avg * (n_trials - len(all_durations))) / n_jobs
            trial_pbar.set_postfix_str(
                f"ETA: {eta / 60:.1f}m, Avg: {avg:.1f}s, Last: {dur:.1f}s"
            )

        # try to set threshold (no-op if not enough data or already set)
        maybe_set_runtime_threshold(study_, warmup_target, include_pruned=True)

    download_data(config["dataset"]["name"])

    with tqdm(
        total=n_trials,
        desc=f"Tuning {study_name}",
        position=1,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}{postfix}]",
    ) as trial_pbar:
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[
                MaxTrialsCallback(n_trials, states=[TrialState.COMPLETE]),
                trial_complete_callback,
            ],
        )


def run_all_studies(config: dict, main_study_name: str, db_url: str):
    surrogates = config["surrogates"]
    global_params = config.get("global_optuna_params", {})

    total_sub_studies = len(surrogates)
    with tqdm(
        total=total_sub_studies, desc="Overall Surrogates", position=0, leave=True
    ) as arch_pbar:
        if config.get("multi_objective", False):
            print(
                "⚠️ Multi-objective mode enabled: using NSGA-II sampler and disabling pruning."
            )

        for surr in surrogates:
            local = surr.get("optuna_params", {})
            for name, opts in global_params.items():
                if name in local:
                    print(
                        f"⚠️ Hyperparameter '{name}' defined globally and locally for {surr['name']}; using local."
                    )
                else:
                    local[name] = opts
            surr["optuna_params"] = local

            arch_name = surr["name"]
            study_name = f"{main_study_name}_{arch_name.lower()}"
            arch_pbar.set_postfix({"study": study_name})

            trials = surr.get("trials", config.get("trials", None))
            sub_config = {
                "batch_size": surr["batch_size"],
                "dataset": config["dataset"],
                "devices": config["devices"],
                "epochs": surr["epochs"],
                "n_trials": trials,
                "seed": config["seed"],
                "surrogate": {"name": arch_name},
                "optuna_params": surr["optuna_params"],
                "prune": config.get("prune", True),
                "optuna_logging": config.get("optuna_logging", False),
                "use_optimal_params": config.get("use_optimal_params", False),
                "multi_objective": config.get("multi_objective", False),
                "population_size": config.get("population_size", 50),
                "target_percentile": config.get("target_percentile", 0.95),
            }

            run_single_study(sub_config, study_name, db_url)
            arch_pbar.update(1)
            arch_pbar.set_postfix({"done": study_name})


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Optuna tuning studies.")
    parser.add_argument(
        "--config",
        type=str,
        default="codes/tune/optuna_config.yaml",
        help="Path to master optuna_config.yaml",
    )
    return parser.parse_args()


def main():
    nice_print("Starting Optuna tuning")
    args = parse_arguments()

    master_cfg_path = Path(args.config).resolve()
    if not master_cfg_path.exists():
        print(f"Config file not found: {master_cfg_path}")
        sys.exit(1)

    config = load_yaml_config(str(master_cfg_path))
    prepare_workspace(master_cfg_path, config)
    tuning_id = config["tuning_id"]

    # Initialize DB (remote/local)
    db_url = initialize_optuna_database(config, study_folder_name=tuning_id)

    # If overwriting, delete Optuna studies
    delete_studies_if_requested(config, tuning_id, db_url)

    # Run
    if "surrogates" in config:
        run_all_studies(config, tuning_id, db_url)
    else:
        run_single_study(config, tuning_id, db_url)

    nice_print("Optuna tuning completed!")


if __name__ == "__main__":
    main()
