from datetime import timedelta

from tqdm import tqdm

from train import parallel_training, sequential_training, train_surrogate
from utils import (
    check_training_status,
    load_and_save_config,
    load_task_list,
    nice_print,
    save_task_list,
)


def main():
    config = load_and_save_config()
    task_list_filepath = check_training_status(config["training_id"])
    tasks = load_task_list(task_list_filepath)

    if not tasks:
        tasks = []

        nice_print("Starting training")

        for surr_name in config["surrogates"]:
            tasks += train_surrogate(config, surr_name)

        save_task_list(tasks, task_list_filepath)

    if tasks:  # Only proceed if there are tasks to complete
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

    else:
        print("No tasks remaining. Training is already completed.")


if __name__ == "__main__":
    main()
