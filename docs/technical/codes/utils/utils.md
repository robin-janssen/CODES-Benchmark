# Utils

[Codes-benchmark Index](../../README.md#codes-benchmark-index) / [Codes](../index.md#codes) / [Utils](./index.md#utils) / Utils

> Auto-generated documentation for [codes.utils.utils](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py) module.

- [Utils](#utils)
  - [check_training_status](#check_training_status)
  - [create_model_dir](#create_model_dir)
  - [get_progress_bar](#get_progress_bar)
  - [load_and_save_config](#load_and_save_config)
  - [load_task_list](#load_task_list)
  - [make_description](#make_description)
  - [nice_print](#nice_print)
  - [read_yaml_config](#read_yaml_config)
  - [save_task_list](#save_task_list)
  - [set_random_seeds](#set_random_seeds)
  - [time_execution](#time_execution)
  - [worker_init_fn](#worker_init_fn)

## check_training_status

[Show source in utils.py:221](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L221)

Check if the training is already completed by looking for a completion marker file.
If the training is not complete, compare the configurations and ask for a confirmation if there are differences.

#### Arguments

- `config` *dict* - The configuration dictionary.

#### Returns

- `str` - The path to the task list file.
- `bool` - Whether to copy the configuration file.

#### Signature

```python
def check_training_status(config: dict) -> tuple[str, bool]: ...
```



## create_model_dir

[Show source in utils.py:43](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L43)

Create a directory based on a unique identifier inside a specified subfolder of the base directory.

#### Arguments

- `base_dir` *str* - The base directory where the subfolder and unique directory will be created.
- `subfolder` *str* - The subfolder inside the base directory to include before the unique directory.
- `unique_id` *str* - A unique identifier to be included in the directory name.

#### Returns

- `str` - The path of the created unique directory within the specified subfolder.

#### Signature

```python
def create_model_dir(
    base_dir: str = ".", subfolder: str = "trained", unique_id: str = ""
) -> str: ...
```



## get_progress_bar

[Show source in utils.py:159](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L159)

Create a progress bar with a specific description.

#### Arguments

- `tasks` *list* - The list of tasks to be executed.

#### Returns

- `tqdm` - The created progress bar.

#### Signature

```python
def get_progress_bar(tasks: list) -> tqdm: ...
```



## load_and_save_config

[Show source in utils.py:66](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L66)

Load configuration from a YAML file and save a copy to the specified directory.

#### Arguments

- `config_path` *str* - The path to the configuration YAML file.
- `save` *bool* - Whether to save a copy of the configuration file. Default is True.

#### Returns

- `dict` - The loaded configuration dictionary.

#### Signature

```python
def load_and_save_config(
    config_path: str = "config.yaml", save: bool = True
) -> dict: ...
```



## load_task_list

[Show source in utils.py:203](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L203)

Load a list of tasks from a JSON file.

#### Arguments

- `filepath` *str* - The path to the JSON file.

#### Returns

- `list` - The loaded list of tasks

#### Signature

```python
def load_task_list(filepath: str | None) -> list: ...
```



## make_description

[Show source in utils.py:134](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L134)

Create a formatted description for the progress bar that ensures consistent alignment.

#### Arguments

- `mode` *str* - The benchmark mode (e.g., "accuracy", "interpolation", "extrapolation", "sparse", "UQ").
- `device` *str* - The device to use for training (e.g., 'cuda:0').
- `metric` *str* - The specific metric for the mode (e.g., interval, cutoff, factor, batch size).
- `surrogate_name` *str* - The name of the surrogate model.

#### Returns

- `str` - A formatted description string for the progress bar.

#### Signature

```python
def make_description(
    mode: str, device: str, metric: str, surrogate_name: str
) -> str: ...
```



## nice_print

[Show source in utils.py:111](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L111)

Print a message in a nicely formatted way with a fixed width.

#### Arguments

- `message` *str* - The message to print.
- `width` *int* - The width of the printed box. Default is 80.

#### Signature

```python
def nice_print(message: str, width: int = 80) -> None: ...
```



## read_yaml_config

[Show source in utils.py:15](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L15)

#### Signature

```python
def read_yaml_config(config_path): ...
```



## save_task_list

[Show source in utils.py:191](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L191)

Save a list of tasks to a JSON file.

#### Arguments

- `tasks` *list* - The list of tasks to save.
- `filepath` *str* - The path to the JSON file.

#### Signature

```python
def save_task_list(tasks: list, filepath: str) -> None: ...
```



## set_random_seeds

[Show source in utils.py:96](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L96)

Set random seeds for reproducibility.

#### Arguments

- `seed` *int* - The random seed to set.

#### Signature

```python
def set_random_seeds(seed: int): ...
```



## time_execution

[Show source in utils.py:21](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L21)

Decorator to time the execution of a function and store the duration
as an attribute of the function.

#### Arguments

- `func` *callable* - The function to be timed.

#### Signature

```python
def time_execution(func): ...
```



## worker_init_fn

[Show source in utils.py:179](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/utils.py#L179)

Initialize the random seed for each worker in PyTorch DataLoader.

#### Arguments

- `worker_id` *int* - The worker ID.

#### Signature

```python
def worker_init_fn(worker_id): ...
```