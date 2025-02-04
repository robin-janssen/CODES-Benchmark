import argparse
import os
import sys

import optuna

from codes.tune.optuna_fcts import (
    initialize_optuna_database,
    load_yaml_config,
    run_all_studies,
    run_single_study,
)
from codes.tune.optuna_mpi import run_mpi_optuna_tuning
from codes.utils import nice_print, setup_logging


def is_mpi_mode():
    return "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ


def run_standard(config: dict, study_name: str, study_folder_name: str):
    db_url = initialize_optuna_database(config, study_folder_name)
    if "surrogates" in config:
        run_all_studies(config, study_name, db_url)
    else:
        run_single_study(config, study_name, db_url)


def run_mpi(config: dict, main_study_name: str, study_folder_name: str):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        db_url = initialize_optuna_database(config, study_folder_name)
    else:
        db_url = None
    comm.Barrier()
    db_url = comm.bcast(db_url, root=0)
    print("db_url", db_url)

    surrogates = config.get("surrogates", None)
    if surrogates is None:
        if rank == 0:
            nice_print(f"Starting MPI Optuna tuning for study: {main_study_name}")
        run_mpi_optuna_tuning(config, main_study_name, db_url)
    else:
        for s in surrogates:
            arch_name = s["name"]
            study_name = f"{main_study_name}_{arch_name.lower()}"
            sub_config = {
                "batch_size": s["batch_size"],
                "dataset": config["dataset"],
                "devices": config["devices"],
                "epochs": s["epochs"],
                "n_trials": s.get("trials", config.get("trials", None)),
                "seed": config["seed"],
                "surrogate": {"name": arch_name},
                "optuna_params": s["optuna_params"],
                "prune": config.get("prune", True),
                "optuna_logging": config.get("optuna_logging", False),
                "use_optimal_params": config.get("use_optimal_params", False),
            }
            if rank == 0:
                nice_print(f"Starting MPI tuning for study: {study_name}")
            run_mpi_optuna_tuning(sub_config, study_name, db_url)
            if rank == 0:
                nice_print("Completed study:", study_name)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run Optuna tuning studies locally.")
    parser.add_argument(
        "--study_name", type=str, default="lotkavolterra2lr", help="Study identifier."
    )
    args = parser.parse_args()

    config_path = os.path.join("optuna_runs", args.study_name, "optuna_config.yaml")
    study_folder_name = os.path.basename(os.path.dirname(config_path))
    config = load_yaml_config(config_path)

    if not config.get("optuna_logging", False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if not is_mpi_mode():
            print("Optuna logging disabled. No intermediate results will be printed.")

    if is_mpi_mode():
        run_mpi(config, args.study_name, study_folder_name)
    else:
        run_standard(config, args.study_name, study_folder_name)

    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() == 0:
            nice_print("Optuna tuning completed!")
    except ImportError:
        nice_print("Optuna tuning completed!")


if __name__ == "__main__":
    main()
