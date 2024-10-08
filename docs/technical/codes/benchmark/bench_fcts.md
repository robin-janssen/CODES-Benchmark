# Bench Fcts

[Codes-benchmark Index](../../README.md#codes-benchmark-index) / [Codes](../index.md#codes) / [Benchmark](./index.md#benchmark) / Bench Fcts

> Auto-generated documentation for [codes.benchmark.bench_fcts](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py) module.

- [Bench Fcts](#bench-fcts)
  - [compare_MAE](#compare_mae)
  - [compare_UQ](#compare_uq)
  - [compare_batchsize](#compare_batchsize)
  - [compare_dynamic_accuracy](#compare_dynamic_accuracy)
  - [compare_extrapolation](#compare_extrapolation)
  - [compare_inference_time](#compare_inference_time)
  - [compare_interpolation](#compare_interpolation)
  - [compare_main_losses](#compare_main_losses)
  - [compare_models](#compare_models)
  - [compare_relative_errors](#compare_relative_errors)
  - [compare_sparse](#compare_sparse)
  - [evaluate_UQ](#evaluate_uq)
  - [evaluate_accuracy](#evaluate_accuracy)
  - [evaluate_batchsize](#evaluate_batchsize)
  - [evaluate_compute](#evaluate_compute)
  - [evaluate_dynamic_accuracy](#evaluate_dynamic_accuracy)
  - [evaluate_extrapolation](#evaluate_extrapolation)
  - [evaluate_interpolation](#evaluate_interpolation)
  - [evaluate_sparse](#evaluate_sparse)
  - [run_benchmark](#run_benchmark)
  - [tabular_comparison](#tabular_comparison)
  - [time_inference](#time_inference)

## compare_MAE

[Show source in bench_fcts.py:863](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L863)

Compare the MAE of different surrogate models over the course of training.

#### Arguments

- `metrics` *dict* - dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def compare_MAE(metrics: dict, config: dict) -> None: ...
```



## compare_UQ

[Show source in bench_fcts.py:1113](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L1113)

Compare the uncertainty quantification (UQ) metrics of different surrogate models.

#### Arguments

- `all_metrics` *dict* - dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def compare_UQ(all_metrics: dict, config: dict) -> None: ...
```



## compare_batchsize

[Show source in bench_fcts.py:1082](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L1082)

Compare the batch size training errors of different surrogate models.

#### Arguments

- `all_metrics` *dict* - dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def compare_batchsize(all_metrics: dict, config: dict) -> None: ...
```



## compare_dynamic_accuracy

[Show source in bench_fcts.py:960](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L960)

Compare the gradients of different surrogate models.

#### Arguments

- `metrics` *dict* - dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def compare_dynamic_accuracy(metrics: dict, config: dict) -> None: ...
```



## compare_extrapolation

[Show source in bench_fcts.py:1021](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L1021)

Compare the extrapolation errors of different surrogate models.

#### Arguments

- `all_metrics` *dict* - dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def compare_extrapolation(all_metrics: dict, config: dict) -> None: ...
```



## compare_inference_time

[Show source in bench_fcts.py:928](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L928)

Compare the mean inference time of different surrogate models.

#### Arguments

metrics (dict[str, dict]): dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.
- `save` *bool, optional* - Whether to save the plot. Defaults to True.

#### Returns

None

#### Signature

```python
def compare_inference_time(
    metrics: dict[str, dict], config: dict, save: bool = True
) -> None: ...
```



## compare_interpolation

[Show source in bench_fcts.py:991](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L991)

Compare the interpolation errors of different surrogate models.

#### Arguments

- `all_metrics` *dict* - dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def compare_interpolation(all_metrics: dict, config: dict) -> None: ...
```



## compare_main_losses

[Show source in bench_fcts.py:824](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L824)

Compare the training and test losses of the main models for different surrogate models.

#### Arguments

- `metrics` *dict* - dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def compare_main_losses(metrics: dict, config: dict) -> None: ...
```



## compare_models

[Show source in bench_fcts.py:776](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L776)

#### Signature

```python
def compare_models(metrics: dict, config: dict): ...
```



## compare_relative_errors

[Show source in bench_fcts.py:897](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L897)

Compare the relative errors over time for different surrogate models.

#### Arguments

- `metrics` *dict* - dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def compare_relative_errors(metrics: dict[str, dict], config: dict) -> None: ...
```



## compare_sparse

[Show source in bench_fcts.py:1051](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L1051)

Compare the sparse training errors of different surrogate models.

#### Arguments

- `all_metrics` *dict* - dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def compare_sparse(all_metrics: dict, config: dict) -> None: ...
```



## evaluate_UQ

[Show source in bench_fcts.py:695](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L695)

Evaluate the uncertainty quantification (UQ) performance of the surrogate model.

#### Arguments

- `model` - Instance of the surrogate model class.
- `surr_name` *str* - The name of the surrogate model.
- `test_loader` *DataLoader* - The DataLoader object containing the test data.
- `timesteps` *np.ndarray* - The timesteps array.
- `conf` *dict* - The configuration dictionary.
- `labels` *list, optional* - The labels for the chemical species.

#### Returns

- `dict` - A dictionary containing UQ metrics.

#### Signature

```python
def evaluate_UQ(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    conf: dict,
    labels: list[str] | None = None,
) -> dict[str, Any]: ...
```



## evaluate_accuracy

[Show source in bench_fcts.py:173](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L173)

Evaluate the accuracy of the surrogate model.

#### Arguments

- `model` - Instance of the surrogate model class.
- `surr_name` *str* - The name of the surrogate model.
- `test_loader` *DataLoader* - The DataLoader object containing the test data.
- `conf` *dict* - The configuration dictionary.
- `labels` *list, optional* - The labels for the chemical species.

#### Returns

- `dict` - A dictionary containing accuracy metrics.

#### Signature

```python
def evaluate_accuracy(
    model,
    surr_name: str,
    test_loader: DataLoader,
    conf: dict,
    labels: list | None = None,
) -> dict[str, Any]: ...
```



## evaluate_batchsize

[Show source in bench_fcts.py:632](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L632)

Evaluate the performance of the surrogate model with different batch sizes.

#### Arguments

- `model` - Instance of the surrogate model class.
- `surr_name` *str* - The name of the surrogate model.
- `test_loader` *DataLoader* - The DataLoader object containing the test data.
- `timesteps` *np.ndarray* - The timesteps array.
- `conf` *dict* - The configuration dictionary.

#### Returns

- `dict` - A dictionary containing batch size training metrics.

#### Signature

```python
def evaluate_batchsize(
    model, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: dict
) -> dict[str, Any]: ...
```



## evaluate_compute

[Show source in bench_fcts.py:382](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L382)

Evaluate the computational resource requirements of the surrogate model.

#### Arguments

- `model` - Instance of the surrogate model class.
- `surr_name` *str* - The name of the surrogate model.
- `test_loader` *DataLoader* - The DataLoader object containing the test data.
- `conf` *dict* - The configuration dictionary.

#### Returns

- `dict` - A dictionary containing model complexity metrics.

#### Signature

```python
def evaluate_compute(
    model, surr_name: str, test_loader: DataLoader, conf: dict
) -> dict[str, Any]: ...
```



## evaluate_dynamic_accuracy

[Show source in bench_fcts.py:248](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L248)

Evaluate the gradients of the surrogate model.

#### Arguments

- `model` - Instance of the surrogate model class.
- `surr_name` *str* - The name of the surrogate model.
- `test_loader` *DataLoader* - The DataLoader object containing the test data.
- `conf` *dict* - The configuration dictionary.

#### Returns

- `dict` - A dictionary containing gradients metrics.

#### Signature

```python
def evaluate_dynamic_accuracy(
    model,
    surr_name: str,
    test_loader: DataLoader,
    conf: dict,
    species_names: list = None,
) -> dict: ...
```



## evaluate_extrapolation

[Show source in bench_fcts.py:485](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L485)

Evaluate the extrapolation performance of the surrogate model.

#### Arguments

- `model` - Instance of the surrogate model class.
- `surr_name` *str* - The name of the surrogate model.
- `test_loader` *DataLoader* - The DataLoader object containing the test data.
- `timesteps` *np.ndarray* - The timesteps array.
- `conf` *dict* - The configuration dictionary.

#### Returns

- `dict` - A dictionary containing extrapolation metrics.

#### Signature

```python
def evaluate_extrapolation(
    model, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: dict
) -> dict[str, Any]: ...
```



## evaluate_interpolation

[Show source in bench_fcts.py:417](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L417)

Evaluate the interpolation performance of the surrogate model.

#### Arguments

- `model` - Instance of the surrogate model class.
- `surr_name` *str* - The name of the surrogate model.
- `test_loader` *DataLoader* - The DataLoader object containing the test data.
- `timesteps` *np.ndarray* - The timesteps array.
- `conf` *dict* - The configuration dictionary.

#### Returns

- `dict` - A dictionary containing interpolation metrics.

#### Signature

```python
def evaluate_interpolation(
    model, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: dict
) -> dict[str, Any]: ...
```



## evaluate_sparse

[Show source in bench_fcts.py:554](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L554)

Evaluate the performance of the surrogate model with sparse training data.

#### Arguments

- `model` - Instance of the surrogate model class.
- `surr_name` *str* - The name of the surrogate model.
- `test_loader` *DataLoader* - The DataLoader object containing the test data.
- `n_train_samples` *int* - The number of training samples in the full dataset.
- `conf` *dict* - The configuration dictionary.

#### Returns

- `dict` - A dictionary containing sparse training metrics.

#### Signature

```python
def evaluate_sparse(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    n_train_samples: int,
    conf: dict,
) -> dict[str, Any]: ...
```



## run_benchmark

[Show source in bench_fcts.py:47](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L47)

Run benchmarks for a given surrogate model.

#### Arguments

- `surr_name` *str* - The name of the surrogate model to benchmark.
- `surrogate_class` - The class of the surrogate model.
- `conf` *dict* - The configuration dictionary.

#### Returns

- `dict` - A dictionary containing all relevant metrics for the given model.

#### Signature

```python
def run_benchmark(surr_name: str, surrogate_class, conf: dict) -> dict[str, Any]: ...
```



## tabular_comparison

[Show source in bench_fcts.py:1146](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L1146)

Compare the metrics of different surrogate models in a tabular format.

#### Arguments

- `all_metrics` *dict* - dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def tabular_comparison(all_metrics: dict, config: dict) -> None: ...
```



## time_inference

[Show source in bench_fcts.py:326](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_fcts.py#L326)

Time the inference of the surrogate model.

#### Arguments

- `model` - Instance of the surrogate model class.
- `surr_name` *str* - The name of the surrogate model.
- `test_loader` *DataLoader* - The DataLoader object containing the test data.
- `timesteps` *np.ndarray* - The timesteps array.
- `conf` *dict* - The configuration dictionary.
- `n_test_samples` *int* - The number of test samples.
- `n_runs` *int, optional* - Number of times to run the inference for timing.

#### Returns

- `dict` - A dictionary containing timing metrics.

#### Signature

```python
def time_inference(
    model,
    surr_name: str,
    test_loader: DataLoader,
    conf: dict,
    n_test_samples: int,
    n_runs: int = 5,
) -> dict[str, Any]: ...
```