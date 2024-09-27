# Bench Utils

[CODES Index](../../README.md#codes-index) / [Codes](../index.md#codes) / [Benchmark](./index.md#benchmark) / Bench Utils

> Auto-generated documentation for [codes.benchmark.bench_utils](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py) module.

- [Bench Utils](#bench-utils)
  - [check_benchmark](#check_benchmark)
  - [check_surrogate](#check_surrogate)
  - [clean_metrics](#clean_metrics)
  - [convert_dict_to_scientific_notation](#convert_dict_to_scientific_notation)
  - [convert_to_standard_types](#convert_to_standard_types)
  - [count_trainable_parameters](#count_trainable_parameters)
  - [discard_numpy_entries](#discard_numpy_entries)
  - [flatten_dict](#flatten_dict)
  - [format_seconds](#format_seconds)
  - [format_time](#format_time)
  - [get_model_config](#get_model_config)
  - [get_required_models_list](#get_required_models_list)
  - [get_surrogate](#get_surrogate)
  - [load_model](#load_model)
  - [make_comparison_csv](#make_comparison_csv)
  - [measure_memory_footprint](#measure_memory_footprint)
  - [read_yaml_config](#read_yaml_config)
  - [write_metrics_to_yaml](#write_metrics_to_yaml)

## check_benchmark

[Show source in bench_utils.py:43](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L43)

Check whether there are any configuration issues with the benchmark.

#### Arguments

- `conf` *dict* - The configuration dictionary.

#### Raises

- `FileNotFoundError` - If the training ID directory is missing or if the .yaml file is missing.
- `ValueError` - If the configuration is missing required keys or the values do not match the training configuration.

#### Signature

```python
def check_benchmark(conf: dict) -> None: ...
```



## check_surrogate

[Show source in bench_utils.py:16](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L16)

Check whether the required models for the benchmark are present in the expected directories.

#### Arguments

- `surrogate` *str* - The name of the surrogate model to check.
- `conf` *dict* - The configuration dictionary.

#### Raises

- `FileNotFoundError` - If any required models are missing.

#### Signature

```python
def check_surrogate(surrogate: str, conf: dict) -> None: ...
```



## clean_metrics

[Show source in bench_utils.py:385](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L385)

Clean the metrics dictionary to remove problematic entries.

#### Arguments

- `metrics` *dict* - The benchmark metrics.
- `conf` *dict* - The configuration dictionary.

#### Returns

- `dict` - The cleaned metrics dictionary.

#### Signature

```python
def clean_metrics(metrics: dict, conf: dict) -> dict: ...
```



## convert_dict_to_scientific_notation

[Show source in bench_utils.py:535](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L535)

Convert all numerical values in a dictionary to scientific notation.

#### Arguments

- `d` *dict* - The input dictionary.

#### Returns

- `dict` - The dictionary with numerical values in scientific notation.

#### Signature

```python
def convert_dict_to_scientific_notation(d: dict, precision: int = 8) -> dict: ...
```



## convert_to_standard_types

[Show source in bench_utils.py:336](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L336)

Recursively convert data to standard types that can be serialized to YAML.

#### Arguments

- `data` - The data to convert.

#### Returns

The converted data.

#### Signature

```python
def convert_to_standard_types(data): ...
```



## count_trainable_parameters

[Show source in bench_utils.py:257](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L257)

Count the number of trainable parameters in the model.

#### Arguments

- `model` *torch.nn.Module* - The PyTorch model.

#### Returns

- `int` - The number of trainable parameters.

#### Signature

```python
def count_trainable_parameters(model: torch.nn.Module) -> int: ...
```



## discard_numpy_entries

[Show source in bench_utils.py:360](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L360)

Recursively remove dictionary entries that contain NumPy arrays.

#### Arguments

- `d` *dict* - The input dictionary.

#### Returns

- `dict` - A new dictionary without entries containing NumPy arrays.

#### Signature

```python
def discard_numpy_entries(d: dict) -> dict: ...
```



## flatten_dict

[Show source in bench_utils.py:513](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L513)

Flatten a nested dictionary.

#### Arguments

- `d` *dict* - The dictionary to flatten.
- `parent_key` *str* - The base key string.
- `sep` *str* - The separator between keys.

#### Returns

- `dict` - Flattened dictionary with composite keys.

#### Signature

```python
def flatten_dict(d: dict, parent_key: str = "", sep: str = " - ") -> dict: ...
```



## format_seconds

[Show source in bench_utils.py:497](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L497)

Format a duration given in seconds as hh:mm:ss.

#### Arguments

- `seconds` *int* - The duration in seconds.

#### Returns

- `str` - The formatted duration string.

#### Signature

```python
def format_seconds(seconds: int) -> str: ...
```



## format_time

[Show source in bench_utils.py:472](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L472)

Format mean and std time consistently in ns, µs, ms, or s.

#### Arguments

- `mean_time` - The mean time.
- `std_time` - The standard deviation of the time.

#### Returns

- `str` - The formatted time string.

#### Signature

```python
def format_time(mean_time, std_time): ...
```



## get_model_config

[Show source in bench_utils.py:614](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L614)

Get the model configuration for a specific surrogate model from the dataset folder.
Returns an empty dictionary if the configuration file is not found.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `config` *dict* - The configuration dictionary.

#### Returns

- `dict` - The model configuration dictionary.

#### Signature

```python
def get_model_config(surr_name: str, config: dict) -> dict: ...
```



## get_required_models_list

[Show source in bench_utils.py:171](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L171)

Generate a list of required models based on the configuration settings.

#### Arguments

- `surrogate` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.

#### Returns

- `list` - A list of required model names.

#### Signature

```python
def get_required_models_list(surrogate: str, conf: dict) -> list: ...
```



## get_surrogate

[Show source in bench_utils.py:455](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L455)

Check if the surrogate model exists.

#### Arguments

- `surrogate_name` *str* - The name of the surrogate model.

#### Returns

SurrogateModel | None: The surrogate model class if it exists, otherwise None.

#### Signature

```python
def get_surrogate(surrogate_name: str) -> SurrogateModel | None: ...
```



## load_model

[Show source in bench_utils.py:234](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L234)

Load a trained surrogate model.

#### Arguments

- `model` - Instance of the surrogate model class.
- `training_id` *str* - The training identifier.
- `surr_name` *str* - The name of the surrogate model.
- `model_identifier` *str* - The identifier of the model (e.g., 'main').

#### Returns

The loaded surrogate model.

#### Signature

```python
def load_model(
    model, training_id: str, surr_name: str, model_identifier: str
) -> torch.nn.Module: ...
```



## make_comparison_csv

[Show source in bench_utils.py:551](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L551)

Generate a CSV file comparing metrics for different surrogate models.

#### Arguments

- `metrics` *dict* - Dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def make_comparison_csv(metrics: dict, config: dict) -> None: ...
```



## measure_memory_footprint

[Show source in bench_utils.py:270](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L270)

Measure the memory footprint of the model during the forward and backward pass.

#### Arguments

- `model` *torch.nn.Module* - The PyTorch model.
- `inputs` *tuple* - The input data for the model.
- `conf` *dict* - The configuration dictionary.
- `surr_name` *str* - The name of the surrogate model.

#### Returns

- `dict` - A dictionary containing memory footprint measurements.

#### Signature

```python
def measure_memory_footprint(model: torch.nn.Module, inputs: tuple) -> dict: ...
```



## read_yaml_config

[Show source in bench_utils.py:219](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L219)

Read the YAML configuration file.

#### Arguments

- `config_path` *str* - Path to the YAML configuration file.

#### Returns

- `dict` - The configuration dictionary.

#### Signature

```python
def read_yaml_config(config_path: str) -> dict: ...
```



## write_metrics_to_yaml

[Show source in bench_utils.py:426](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_utils.py#L426)

Write the benchmark metrics to a YAML file.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `metrics` *dict* - The benchmark metrics.

#### Signature

```python
def write_metrics_to_yaml(surr_name: str, conf: dict, metrics: dict) -> None: ...
```