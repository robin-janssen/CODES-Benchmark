import os
import threading
from queue import Queue
from threading import Thread

from tqdm import tqdm

from codes.benchmark.bench_utils import get_model_config, get_surrogate
from codes.utils import (
    check_and_load_data,
    determine_batch_size,
    get_data_subset,
    get_progress_bar,
    load_and_save_config,
    load_task_list,
    make_description,
    save_task_list,
    set_random_seeds,
)


class DummyLock:
    def acquire(self):
        pass

    def release(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def train_and_save_model(
    surr_name: str,
    mode: str,
    metric: int,
    training_id: str,
    seed: int | None = None,
    epochs: int | None = None,
    device: str = "cpu",
    position: int = 1,
    threadlock: threading.Lock = DummyLock(),
):
    """
    Train and save a model for a specific benchmark mode. The parameters are determined
    by the task(s) which is created from the config file.
    Threadlock is used to make the training deterministic. In the single-threaded case,
    uses a dummy lock instead.

    Args:
        surr_name (str): The name of the surrogate model.
        mode (str): The benchmark mode (e.g. "main", "interpolation", "extrapolation").
        metric (int): The metric for the benchmark mode.
        training_id (str): The training ID for the current training session.
        seed (int, optional): The random seed for the training. Defaults to None.
        epochs (int, optional): The number of epochs for the training. Defaults to None.
        device (str, optional): The device for the training. Defaults to "cpu".
        position (int, optional): The position of the model in the task list. Defaults to 1.
        threadlock (threading.Lock, optional): A lock to prevent threading issues with PyTorch. Defaults to None.
    """
    config_path = f"trained/{training_id}/config.yaml"
    config = load_and_save_config(config_path, save=False)

    # Load full data
    full_train_data, full_test_data, _, timesteps, _, data_params, _ = (
        check_and_load_data(
            config["dataset"]["name"],
            verbose=False,
            log=config["dataset"]["log10_transform"],
            normalisation_mode=config["dataset"]["normalise"],
            tolerance=config["dataset"]["tolerance"],
        )
    )

    # Get the appropriate data subset
    train_data, test_data, timesteps = get_data_subset(
        full_train_data, full_test_data, timesteps, mode, metric
    )

    _, n_timesteps, n_quantities = train_data.shape

    # # Replace timesteps with dummy timesteps between 0 and 1
    # timesteps = np.linspace(0, 1, n_timesteps)

    # Get the surrogate class
    surrogate_class = get_surrogate(surr_name)
    model_config = get_model_config(surr_name, config)

    # Apply threadlock to avoid issues with PyTorch
    with threadlock:
        set_random_seeds(seed, device)
        model = surrogate_class(device, n_quantities, n_timesteps, model_config)
    model.normalisation = data_params
    surr_idx = config["surrogates"].index(surr_name)

    batch_size = determine_batch_size(config, surr_idx, mode, metric)

    with threadlock:
        set_random_seeds(seed, device)
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
        epochs=epochs,
        position=position,
        description=description,
    )

    # Save the model (making the name lowercase and removing any underscores)
    model_name = f"{surr_name.lower()}_{mode}_{str(metric)}".strip("_")
    model_name = model_name.replace("__", "_")
    base_dir = os.path.join(os.getcwd(), "trained")
    model.save(
        model_name=model_name,
        training_id=config["training_id"],
        base_dir=base_dir,
    )


def create_task_list_for_surrogate(config, surr_name: str) -> list:
    """
    Creates a list of training tasks for a specific surrogate model based on the
    configuration file.

    Args:
        config (dict): The configuration dictionary taken from the config file.
        surr_name (str): The name of the surrogate model.

    Returns:
        list: A list of training tasks for the surrogate model.
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
    errors_encountered: list[bool],
    threadlock: threading.Lock,
):
    """
    Worker function to process tasks from the task queue on the given device.

    Args:
        task_queue (Queue): The in-memory queue containing the training tasks.
        device (str): The device to use for training.
        device_idx (int): The index of the device in the device list.
        overall_progress_bar (tqdm): The overall progress bar for the training.
        task_list_filepath (str): The filepath to the JSON task list.
        errors_encountered (list[bool]): A shared mutable flag array indicating if an error has occurred
                                         (True if at least one task failed).
        threadlock (threading.Lock): A lock to prevent threading issues with PyTorch.
    """
    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()  # Remove the task from the in-memory queue
            train_and_save_model(
                *task, device=device, position=device_idx + 1, threadlock=threadlock
            )

            # Mark that we have successfully processed this task
            task_queue.task_done()
            overall_progress_bar.update(1)
            current_list = load_task_list(task_list_filepath)
            task_as_list = list(task)

            try:
                current_list.remove(task_as_list)
            except ValueError:
                pass

            save_task_list(current_list, task_list_filepath)

        except Exception as e:
            tqdm.write(f"Exception for task {task[:3]}: {e}")
            # Mark this task as "done" for the queue, so the loop can move on
            task_queue.task_done()
            overall_progress_bar.update(1)

            # Flag that at least one task has failed
            errors_encountered[0] = True


def parallel_training(tasks, device_list, task_list_filepath: str):
    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)

    errors_encountered = [False]
    overall_progress_bar = get_progress_bar(tasks)

    threads = []
    threadlock = threading.Lock()
    for i, device in enumerate(device_list):
        thread = Thread(
            target=worker,
            args=(
                task_queue,
                device,
                i,
                overall_progress_bar,
                task_list_filepath,
                errors_encountered,
                threadlock,
            ),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    overall_progress_bar.close()
    elapsed_time = overall_progress_bar.format_dict["elapsed"]

    # If no errors, mark training done & remove tasks file
    if not errors_encountered[0]:
        with open(
            os.path.join(os.path.dirname(task_list_filepath), "completed.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("Training completed")
        os.remove(task_list_filepath)
    else:
        tqdm.write(
            "Some tasks failed. The task list file was NOT removed.\n"
            "Please fix the error(s) and re-run to complete the remaining tasks."
        )

    return elapsed_time


def sequential_training(tasks, device_list, task_list_filepath: str):
    overall_progress_bar = get_progress_bar(tasks)
    errors_encountered = False
    device = device_list[0]

    for task in tasks:
        try:
            train_and_save_model(*task, device=device)
            overall_progress_bar.update(1)

            # Only remove *this* task from JSON if success
            current_list = load_task_list(task_list_filepath)
            task_as_list = list(task)
            try:
                current_list.remove(task_as_list)
            except ValueError:
                pass
            save_task_list(current_list, task_list_filepath)

        except Exception as e:
            tqdm.write(f"Exception for task {task[:3]}: {e}")
            overall_progress_bar.update(1)
            errors_encountered = True

    elapsed_time = overall_progress_bar.format_dict["elapsed"]
    overall_progress_bar.close()

    if not errors_encountered:
        with open(
            os.path.join(os.path.dirname(task_list_filepath), "completed.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("Training completed")
        os.remove(task_list_filepath)
    else:
        tqdm.write(
            "Some tasks failed. The task list file was NOT removed.\n"
            "Please fix the error(s) and re-run to complete the remaining tasks."
        )

    return elapsed_time
