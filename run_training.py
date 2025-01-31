import os
from argparse import ArgumentParser
from datetime import timedelta
from typing import Dict, List

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import logging

import ray
import torch
from tqdm import tqdm

from codes.train import (
    create_task_list_for_surrogate,
    parallel_training,
    sequential_training,
)
from codes.utils import (
    check_training_status,
    get_progress_bar,
    load_and_save_config,
    load_task_list,
    nice_print,
    save_task_list,
)
from codes.utils.data_utils import download_data


@ray.remote(num_gpus=1)
def ray_train_and_save_model(task: list) -> None:
    """
    Ray remote function to train and save a single model.

    Logs will now appear in the Ray Dashboard.
    """
    from codes.train import train_and_save_model

    if len(task) != 6:
        raise ValueError(f"Expected task list of length 6, got {len(task)}")

    surr_name, mode, metric, training_id, seed, epochs = task
    device = "cuda"
    node_ip = ray.util.get_node_ip_address()  # Get the node where this task is running

    try:
        print(
            f"[{node_ip}] Starting training: {surr_name}, mode: {mode}, metric: {metric}, training_id: {training_id}",
            flush=True,
        )

        train_and_save_model(
            surr_name=surr_name,
            mode=mode,
            metric=metric,
            training_id=training_id,
            seed=seed,
            epochs=epochs,
            device=device,
        )

        print(
            f"[{node_ip}] Completed training: {surr_name}, mode: {mode}, metric: {metric}, training_id: {training_id}",
            flush=True,
        )

    except Exception as e:
        print(
            f"[{node_ip}] Training FAILED: {surr_name}, mode: {mode}, metric: {metric}, training_id: {training_id}. Error: {e}",
            flush=True,
        )
        raise e  # Ensure the task is marked as failed in Ray


def parallel_training_ray(tasks: List, config: Dict) -> float:
    """
    Distributes training tasks across the Ray cluster.

    Args:
        tasks (List): A list of training tasks.
        config (Dict): The configuration dictionary.

    Returns:
        float: The total elapsed training time.
    """
    overall_progress_bar = get_progress_bar(tasks)
    errors_encountered = [False]

    # Submit all tasks to Ray
    futures = [ray_train_and_save_model.remote(task) for task in tasks]

    # Retrieve results and update progress
    for future in futures:
        try:
            ray.get(future)
            overall_progress_bar.update(1)
        except Exception as e:
            nice_print(f"Exception during Ray task execution: {e}")
            errors_encountered[0] = True
            overall_progress_bar.update(1)

    overall_progress_bar.close()
    elapsed_time = overall_progress_bar.format_dict.get("elapsed", 0)

    # Handle task completion
    if not errors_encountered[0]:
        task_list_filepath = config.get("task_list_filepath", "task_list.json")
        with open(
            os.path.join(os.path.dirname(task_list_filepath), "completed.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("Training completed")
        os.remove(task_list_filepath)
    else:
        nice_print(
            "Some tasks failed. The task list file was NOT removed.\n"
            "Please fix the error(s) and re-run to complete the remaining tasks."
        )

    return elapsed_time


def parallel_training_dispatcher(
    tasks: List, device_list: List[str], task_list_filepath: str, config: Dict
) -> float:
    """
    Dispatches training tasks either using Ray or multithreading based on configuration.

    Args:
        tasks (List): A list of training tasks.
        device_list (List[str]): List of available devices.
        task_list_filepath (str): Path to the task list file.
        config (Dict): The configuration dictionary.

    Returns:
        float: The total elapsed training time.
    """
    use_ray = config.get("use_ray", False)
    if use_ray:
        return parallel_training_ray(tasks, config)
    else:
        return parallel_training(tasks, device_list, task_list_filepath)


def main(args):
    """
    Main function to train the models. If the training is already completed, it will
    print a message and exit. Otherwise, it will create a task list for each surrogate
    model and train the models sequentially or in parallel depending on the configuration.

    Args:
        args (Namespace): The command line arguments.
    """
    torch.use_deterministic_algorithms(True)
    config = load_and_save_config(config_path=args.config, save=False)
    download_data(config["dataset"]["name"])
    task_list_filepath, copy_config = check_training_status(config)
    if copy_config:
        load_and_save_config(config_path=args.config, save=True)
    tasks = load_task_list(task_list_filepath)

    if not tasks:
        tasks = []
        nice_print("Starting training")
        for surr_name in config["surrogates"]:
            tasks += create_task_list_for_surrogate(config, surr_name)
        save_task_list(tasks, task_list_filepath)

    device_list = config["devices"]
    device_list = [device_list] if isinstance(device_list, str) else device_list

    # Initialize Ray if use_ray is True
    if config.get("use_ray", False):
        ray.init(address="auto")

    # Decide whether to use Ray or multithreading based on configuration and device count
    if len(device_list) > 1 or config.get("use_ray", False):
        elapsed_time = parallel_training_dispatcher(
            tasks, device_list, task_list_filepath, config
        )
    else:
        tqdm.write(f"Training models sequentially on device {device_list[0]}")
        elapsed_time = sequential_training(tasks, device_list, task_list_filepath)

    print("\n")
    nice_print("Training completed")
    print(f"{len(tasks)} Models saved in /trained/{config['training_id']}/")
    print(f"Total training time: {timedelta(seconds=int(elapsed_time))} \n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", default="config.yaml", type=str, help="Path to the config file."
    )
    args = parser.parse_args()
    main(args)
