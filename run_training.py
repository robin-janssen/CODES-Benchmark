from argparse import ArgumentParser
from datetime import timedelta

from tqdm import tqdm

from data.data_utils import download_data
from train import create_task_list_for_surrogate, parallel_training, sequential_training
from utils import (
    check_training_status,
    load_and_save_config,
    load_task_list,
    nice_print,
    save_task_list,
)


def main(args):
    """
    Main function to train the models. If the training is already completed, it will
    print a message and exit. Otherwise, it will create a task list for each surrogate
    model and train the models sequentially or in parallel depending on the number of
    devices.

    Args:
        args (Namespace): The command line arguments.
    """
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
    if len(device_list) > 1:
        tqdm.write(f"Training models in parallel on devices  : {device_list} \n")
        elapsed_time = parallel_training(tasks, device_list, task_list_filepath)
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
