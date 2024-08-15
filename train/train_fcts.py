import os
from queue import Queue
from threading import Thread

from tqdm import tqdm

from benchmark.bench_utils import get_model_config, get_surrogate
from data import check_and_load_data, get_data_subset
from utils import (
    get_progress_bar,
    load_and_save_config,
    make_description,
    save_task_list,
    set_random_seeds,
)


def train_and_save_model(
    surr_name: str,
    mode: str,
    metric: int,
    training_id: str,
    seed: int | None = None,
    epochs: int | None = None,
    device: str = "cpu",
    position: int = 1,
):
    """
    Train and save a model for a specific benchmark mode.
    """
    config_path = f"trained/{training_id}/config.yaml"
    config = load_and_save_config(config_path, save=False)

    # Set the seed for the training
    if seed is not None:
        set_random_seeds(seed)

    # Load full data
    full_train_data, full_test_data, _, timesteps, _, data_params, _ = (
        check_and_load_data(
            config["dataset"]["name"],
            verbose=False,
            log=config["dataset"]["log10_transform"],
            normalisation_mode=config["dataset"]["normalise"],
        )
    )

    model_config = get_model_config(surr_name, config["dataset"]["name"])

    # Get the appropriate data subset
    train_data, test_data, timesteps = get_data_subset(
        full_train_data, full_test_data, timesteps, mode, metric, config
    )

    n_timesteps = train_data.shape[1]
    n_chemicals = train_data.shape[2]

    # Get the surrogate class
    surrogate_class = get_surrogate(surr_name)

    # Set the device for the model
    model = surrogate_class(device, n_chemicals, n_timesteps, model_config)
    surr_idx = config["surrogates"].index(surr_name)

    # Determine the batch size
    if isinstance(config["batch_size"], list):
        if len(config["batch_size"]) != len(config["surrogates"]):
            raise ValueError(
                "The number of provided batch sizes must match the number of surrogate models."
            )
        else:
            batch_size = config["batch_size"][surr_idx]
    else:
        batch_size = config["batch_size"]
    batch_size = metric if mode == "batch_size" else batch_size
    # epochs = epochs if epochs is not None else config["epochs"]

    train_loader, test_loader, _ = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=None,
        timesteps=timesteps,
        batch_size=batch_size,
        shuffle=True,
    )

    description = make_description(mode, device, str(metric), surr_name)

    # Train the model
    model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        timesteps=timesteps,
        epochs=epochs,
        position=position,
        description=description,
    )

    # Save the model (making the name lowercase and removing any underscores)
    model_name = f"{surr_name.lower()}_{mode}_{str(metric)}".strip("_")
    model_name = model_name.replace("__", "_")
    model.save(
        model_name=model_name,
        training_id=config["training_id"],
        subfolder="trained",
        data_params=data_params,
    )


def train_surrogate(config, surr_name: str):
    """
    Train and save models for different purposes based on the config settings.
    """
    tasks = []
    seed = config["seed"]
    surr_idx = config["surrogates"].index(surr_name)
    id = config["training_id"]
    epochs = (
        config["epochs"][surr_idx]
        if isinstance(config["epochs"], list)
        else config["epochs"]
    )

    tasks.append((surr_name, "main", "", id, seed, epochs))

    if config["interpolation"]["enabled"]:
        mode = "interpolation"
        for interval in config["interpolation"]["intervals"]:
            tasks.append((surr_name, mode, interval, id, seed + interval, epochs))

    if config["extrapolation"]["enabled"]:
        mode = "extrapolation"
        for cutoff in config["extrapolation"]["cutoffs"]:
            tasks.append((surr_name, mode, cutoff, id, seed + cutoff, epochs))

    if config["sparse"]["enabled"]:
        for factor in config["sparse"]["factors"]:
            tasks.append((surr_name, "sparse", factor, id, seed + factor, epochs))

    if config["uncertainty"]["enabled"]:
        n_models = config["uncertainty"]["ensemble_size"]
        for i in range(n_models - 1):
            tasks.append((surr_name, "UQ", i + 1, id, seed + i, epochs))

    if config["batch_scaling"]["enabled"]:
        mode = "batchsize"
        for bs in config["batch_scaling"]["sizes"]:
            tasks.append((surr_name, mode, bs, id, seed + bs, epochs))

    return tasks


def worker(
    task_queue: Queue,
    device: str,
    device_idx: int,
    overall_progress_bar: tqdm,
    task_list_filepath: str,
):
    """
    Worker function to process tasks from the task queue on the given device.
    """
    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()
            train_and_save_model(*task, device, position=device_idx + 1)
            task_queue.task_done()
            overall_progress_bar.update(1)

            # Save the remaining tasks after completing each one
            remaining_tasks = list(task_queue.queue)
            save_task_list(remaining_tasks, task_list_filepath)

        except Exception as e:
            tqdm.write(f"Exception for task {task[:3]}: {e}")
            task_queue.task_done()
            overall_progress_bar.update(1)


def parallel_training(tasks, device_list, task_list_filepath: str):
    """
    Execute the training tasks in parallel across multiple devices.
    """
    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)

    # Create the overall progress bar
    overall_progress_bar = get_progress_bar(tasks)
    threads = []
    for i, device in enumerate(device_list):
        thread = Thread(
            target=worker,
            args=(task_queue, device, i, overall_progress_bar, task_list_filepath),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    overall_progress_bar.close()
    elapsed_time = overall_progress_bar.format_dict["elapsed"]

    # Create a completion marker after all tasks are completed
    with open(
        os.path.join(os.path.dirname(task_list_filepath), "completed.txt"), "w"
    ) as f:
        f.write("Training completed")

    # Remove the task list file
    os.remove(task_list_filepath)

    return elapsed_time


def sequential_training(tasks, device_list, task_list_filepath: str):
    """
    Execute the training tasks sequentially on a single device.
    """
    overall_progress_bar = get_progress_bar(tasks)
    for i, task in enumerate(tasks):
        train_and_save_model(*task, device_list[0])
        overall_progress_bar.update(1)
        remaining_tasks = tasks[i + 1 :]
        save_task_list(remaining_tasks, task_list_filepath)

    elapsed_time = overall_progress_bar.format_dict["elapsed"]
    overall_progress_bar.close()

    # Create a completion marker after all tasks are completed
    with open(
        os.path.join(os.path.dirname(task_list_filepath), "completed.txt"), "w"
    ) as f:
        f.write("Training completed")

    # Remove the task list file
    os.remove(task_list_filepath)

    return elapsed_time
