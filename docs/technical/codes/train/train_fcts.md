# Train Fcts

[Codes-benchmark Index](../../README.md#codes-benchmark-index) / [Codes](../index.md#codes) / [Train](./index.md#train) / Train Fcts

> Auto-generated documentation for [codes.train.train_fcts](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/train/train_fcts.py) module.

- [Train Fcts](#train-fcts)
  - [create_task_list_for_surrogate](#create_task_list_for_surrogate)
  - [parallel_training](#parallel_training)
  - [sequential_training](#sequential_training)
  - [train_and_save_model](#train_and_save_model)
  - [worker](#worker)

## create_task_list_for_surrogate

[Show source in train_fcts.py:122](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/train/train_fcts.py#L122)

Creates a list of training tasks for a specific surrogate model based on the
configuration file.

#### Arguments

- `config` *dict* - The configuration dictionary taken from the config file.
- `surr_name` *str* - The name of the surrogate model.

#### Returns

- `list` - A list of training tasks for the surrogate model.

#### Signature

```python
def create_task_list_for_surrogate(config, surr_name: str) -> list: ...
```



## parallel_training

[Show source in train_fcts.py:207](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/train/train_fcts.py#L207)

Execute the training tasks in parallel on multiple devices.

#### Arguments

- `tasks` *list* - The list of training tasks.
- `device_list` *list* - The list of devices to use for training.
- `task_list_filepath` *str* - The filepath to the task list file.

#### Signature

```python
def parallel_training(tasks, device_list, task_list_filepath: str): ...
```



## sequential_training

[Show source in train_fcts.py:251](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/train/train_fcts.py#L251)

Execute the training tasks sequentially on a single device.

#### Arguments

- `tasks` *list* - The list of training tasks.
- `device_list` *list* - The list of devices to use for training.
- `task_list_filepath` *str* - The filepath to the task list file.

#### Signature

```python
def sequential_training(tasks, device_list, task_list_filepath: str): ...
```



## train_and_save_model

[Show source in train_fcts.py:19](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/train/train_fcts.py#L19)

Train and save a model for a specific benchmark mode. The parameters are determined
by the task(s) which is created from the config file.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `mode` *str* - The benchmark mode (e.g. "main", "interpolation", "extrapolation").
- `metric` *int* - The metric for the benchmark mode.
- `training_id` *str* - The training ID for the current training session.
- `seed` *int, optional* - The random seed for the training. Defaults to None.
- `epochs` *int, optional* - The number of epochs for the training. Defaults to None.
- `device` *str, optional* - The device for the training. Defaults to "cpu".
- `position` *int, optional* - The position of the model in the task list. Defaults to 1.

#### Signature

```python
def train_and_save_model(
    surr_name: str,
    mode: str,
    metric: int,
    training_id: str,
    seed: int | None = None,
    epochs: int | None = None,
    device: str = "cpu",
    position: int = 1,
): ...
```



## worker

[Show source in train_fcts.py:173](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/train/train_fcts.py#L173)

Worker function to process tasks from the task queue on the given device.

#### Arguments

- `task_queue` *Queue* - The task queue containing the training tasks.
- `device` *str* - The device to use for training.
- `device_idx` *int* - The index of the device in the device list.
- `overall_progress_bar` *tqdm* - The overall progress bar for the training.
- `task_list_filepath` *str* - The filepath to the task list file

#### Signature

```python
def worker(
    task_queue: Queue,
    device: str,
    device_idx: int,
    overall_progress_bar: tqdm,
    task_list_filepath: str,
): ...
```