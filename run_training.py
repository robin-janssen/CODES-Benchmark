import os
from argparse import ArgumentParser
from datetime import timedelta

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from tqdm import tqdm

from codes.train import create_task_list_for_surrogate
from codes.utils import (
    check_training_status,
    download_data,
    load_and_save_config,
    load_task_list,
    nice_print,
    save_task_list,
    setup_logging,
)


def is_mpi_mode():
    return (
        "OMPI_COMM_WORLD_SIZE" in os.environ
        or "PMI_SIZE" in os.environ
        or "SLURM_NTASKS" in os.environ
    )


def run_mpi_training(config, args):
    from mpi4py import MPI

    from codes.train import master_mpi, worker_mpi

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        torch.use_deterministic_algorithms(True)
        download_data(config["dataset"]["name"])
        task_list_filepath, copy_config = check_training_status(config)
        if copy_config:
            load_and_save_config(config_path=args.config, save=True)
        tasks = load_task_list(task_list_filepath)
        if not tasks:
            tasks = []
            nice_print("Starting MPI training")
            for s in config["surrogates"]:
                tasks += create_task_list_for_surrogate(config, s)
            save_task_list(tasks, task_list_filepath)
        master_mpi(tasks, size, task_list_filepath)
        if len(tasks) > 0:
            with open(
                os.path.join(os.path.dirname(task_list_filepath), "completed.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("Training completed")
            os.remove(task_list_filepath)
        nice_print("All trainings completed")
    else:
        worker_mpi(rank)


def run_standard_training(config, args):
    torch.use_deterministic_algorithms(True)
    download_data(config["dataset"]["name"])
    task_list_filepath, copy_config = check_training_status(config)
    if copy_config:
        load_and_save_config(config_path=args.config, save=True)
    tasks = load_task_list(task_list_filepath)
    if not tasks:
        tasks = []
        nice_print("Starting training")
        for s in config["surrogates"]:
            tasks += create_task_list_for_surrogate(config, s)
        save_task_list(tasks, task_list_filepath)
    device_list = config["devices"]
    device_list = [device_list] if isinstance(device_list, str) else device_list
    if len(device_list) > 1:
        from codes.train import parallel_training

        tqdm.write(f"Training models in parallel on devices: {device_list}\n")
        elapsed_time = parallel_training(tasks, device_list, task_list_filepath)
    else:
        from codes.train import sequential_training

        tqdm.write(f"Training models sequentially on device {device_list[0]}")
        elapsed_time = sequential_training(tasks, device_list, task_list_filepath)
    print("\n")
    nice_print("Training completed")
    print(f"{len(tasks)} Models saved in /trained/{config['training_id']}/")
    print(f"Total training time: {timedelta(seconds=int(elapsed_time))} \n")


def main(args):
    setup_logging()
    config = load_and_save_config(config_path=args.config, save=False)
    if is_mpi_mode():
        run_mpi_training(config, args)
    else:
        run_standard_training(config, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", default="config.yaml", type=str, help="Path to the config file."
    )
    args = parser.parse_args()
    main(args)
