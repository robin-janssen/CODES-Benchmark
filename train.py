from datetime import timedelta

from tqdm import tqdm

from train import parallel_training, train_and_save_model, train_surrogate
from utils import (
    check_training_status,
    get_progress_bar,
    load_and_save_config,
    load_task_list,
    nice_print,
    save_task_list,
)


def main():
    config = load_and_save_config()
    task_list_filepath = check_training_status(config["training_id"])

    # Load tasks from a previous run if they exist
    tasks = load_task_list(task_list_filepath)

    if not tasks:
        tasks = []

        nice_print("Starting training")

        for surr_name in config["surrogates"]:
            tasks += train_surrogate(config, surr_name)

        # Save the initial task list
        save_task_list(tasks, task_list_filepath)

    if tasks:  # Only proceed if there are tasks to complete
        device_list = config["devices"]
        device_list = [device_list] if isinstance(device_list, str) else device_list
        if len(device_list) > 1:
            tqdm.write(f"Training models in parallel on devices  : {device_list} \n")
            elapsed_time = parallel_training(tasks, device_list, task_list_filepath)
        else:
            tqdm.write(f"Training models sequentially on device {device_list[0]}")
            overall_progress_bar = get_progress_bar(tasks)
            for i, task in enumerate(tasks):
                train_and_save_model(*task, device_list[0])
                overall_progress_bar.update(1)
                remaining_tasks = tasks[i + 1 :]
                save_task_list(remaining_tasks, task_list_filepath)
            elapsed_time = overall_progress_bar.format_dict["elapsed"]
            overall_progress_bar.close()

        print("\n")
        nice_print("Training completed")

        print(f"{len(tasks)} Models saved in /trained/{config['training_id']}/")
        print(f"Total training time: {timedelta(seconds=int(elapsed_time))} \n")

    else:
        print("No tasks remaining. Training is already completed.")


if __name__ == "__main__":
    main()
