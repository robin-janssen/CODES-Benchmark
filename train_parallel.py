import os

# os.environ["TQDM_DISABLE"] = "1"

from queue import Queue
from threading import Thread

from surrogates.surrogate_classes import surrogate_classes
from utils import load_and_save_config, set_random_seeds, nice_print
from data import check_and_load_data, get_data_subset


def train_and_save_model(
    mode: str,
    surrogate_name: str,
    metric: str,
    surrogate_class,
    config,
    seed: int | None = None,
    epochs: int | None = None,
    device: str = "cpu",
):
    """
    Train and save a model for a specific benchmark mode.

    Args:
        mode (str): The benchmark mode (e.g., "accuracy", "interpolation", "extrapolation", "sparse", "UQ").
        surrogate_name (str): The name of the surrogate model.
        metric (str): The specific metric for the mode (e.g., interval, cutoff, factor, batch size).
        surrogate_class: The class of the surrogate model.
        config (dict): The configuration dictionary.
        seed (int): The seed for initializing the model and shuffling the data.
        epochs (int): The number of epochs to train the model.
        device (str): The device to use for training (e.g., 'cuda:0').
    """
    # Set the seed for the training
    if seed is not None:
        set_random_seeds(seed)

    # Set the device for the model
    model = surrogate_class(device=device)

    # Determine the batch size and number of epochs
    batch_size = int(metric) if mode == "batch_size" else config["batch_size"]
    epochs = epochs if epochs is not None else config["epochs"]

    # Load full data
    full_train_data, full_test_data, _, timesteps, _ = check_and_load_data(
        config["dataset"], verbose=False
    )

    # Get the appropriate data subset
    train_data, test_data, timesteps = get_data_subset(
        full_train_data, full_test_data, timesteps, mode, metric, config
    )

    train_loader, test_loader, _ = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=None,
        timesteps=timesteps,
        batch_size=batch_size,
        shuffle=True,
    )

    # Train the model
    model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        timesteps=timesteps,
        epochs=epochs,
    )

    # Save the model (making the name lowercase and removing any underscores)
    model_name = f"{surrogate_name.lower()}_{mode}_{metric}".strip("_")
    model_name = model_name.replace("__", "_")
    model.save(
        model_name=model_name,
        training_id=config["training_id"],
        subfolder="trained",
        dataset_name=config["dataset"],
    )


def train_surrogate(config, surrogate_class, surrogate_name):
    """
    Train and save models for different purposes based on the config settings.

    Args:
        config (dict): The configuration dictionary.
        surrogate_class: The class of the surrogate model to train.
        surrogate_name (str): The name of the surrogate model.
    """

    tasks = []
    seed = config["seed"]
    epochs = config["epochs"]

    if config["accuracy"]:
        tasks.append(
            ("main", surrogate_name, "", surrogate_class, config, seed, epochs)
        )

    if config["interpolation"]["enabled"]:
        for interval in config["interpolation"]["intervals"]:
            tasks.append(
                (
                    "interpolation",
                    surrogate_name,
                    str(interval),
                    surrogate_class,
                    config,
                    seed + interval,
                    epochs,
                )
            )

    if config["extrapolation"]["enabled"]:
        for cutoff in config["extrapolation"]["cutoffs"]:
            tasks.append(
                (
                    "extrapolation",
                    surrogate_name,
                    str(cutoff),
                    surrogate_class,
                    config,
                    seed + cutoff,
                    epochs,
                )
            )

    if config["sparse"]["enabled"]:
        for factor in config["sparse"]["factors"]:
            tasks.append(
                (
                    "sparse",
                    surrogate_name,
                    str(factor),
                    surrogate_class,
                    config,
                    seed + factor,
                    epochs,
                )
            )

    if config["UQ"]["enabled"]:
        n_models = config["UQ"]["n_models"]
        for i in range(n_models - 1):
            tasks.append(
                (
                    "UQ",
                    surrogate_name,
                    str(i + 1),
                    surrogate_class,
                    config,
                    seed + i,
                    epochs,
                )
            )

    if config["batch_scaling"]["enabled"]:
        for batch_size in config["batch_scaling"]["sizes"]:
            tasks.append(
                (
                    "batchsize",
                    surrogate_name,
                    str(batch_size),
                    surrogate_class,
                    config,
                    seed + batch_size,
                    config["batch_scaling"]["epochs"],
                )
            )

    return tasks


def worker(task_queue: Queue, device: str):
    """
    Worker function to process tasks from the task queue on the given device.

    Args:
        task_queue (Queue): The queue containing tasks to be processed.
        device (str): The device to use for processing tasks.
    """
    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()
            print(f"Starting training for task: {task[:3]} on device {device}")
            train_and_save_model(*task, device)
            task_queue.task_done()
            print(f"Completed training for task: {task[:3]}")
        except Exception as e:
            print(f"Exception for task {task[:3]}: {e}")
            task_queue.task_done()


def parallel_training(tasks, device_list):
    """
    Execute the training tasks in parallel across multiple devices.

    Args:
        tasks (list): A list of tasks to execute in parallel.
        device_list (list): A list of devices to use for parallel training.
    """
    os.environ["TQDM_DISABLE"] = "1"

    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)

    threads = []
    for device in device_list:
        thread = Thread(target=worker, args=(task_queue, device))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def main():
    config = load_and_save_config()
    tasks = []
    device_list = config["devices"]
    device_list = [device_list] if isinstance(device_list, str) else device_list

    for surrogate_name in config["surrogates"]:
        if surrogate_name in surrogate_classes:
            nice_print(f"Training surrogate model: {surrogate_name}")
            surrogate_class = surrogate_classes[surrogate_name]
            tasks += train_surrogate(config, surrogate_class, surrogate_name)
        else:
            print(f"Surrogate {surrogate_name} not recognized. Skipping.")

    if len(device_list) > 1:
        parallel_training(tasks, device_list)
    else:
        for task in tasks:
            train_and_save_model(*task, device_list[0])


if __name__ == "__main__":
    main()
