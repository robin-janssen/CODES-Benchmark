# Data Utils

[CODES Index](../../README.md#codes-index) / [Codes](../index.md#codes) / [Utils](./index.md#utils) / Data Utils

> Auto-generated documentation for [codes.utils.data_utils](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/data_utils.py) module.

- [Data Utils](#data-utils)
  - [DatasetError](#dataseterror)
  - [check_and_load_data](#check_and_load_data)
  - [create_dataset](#create_dataset)
  - [create_hdf5_dataset](#create_hdf5_dataset)
  - [download_data](#download_data)
  - [get_data_subset](#get_data_subset)
  - [normalize_data](#normalize_data)

## DatasetError

[Show source in data_utils.py:10](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/data_utils.py#L10)

Error for missing data or dataset or if the data shape is incorrect.

#### Signature

```python
class DatasetError(Exception): ...
```



## check_and_load_data

[Show source in data_utils.py:18](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/data_utils.py#L18)

Check the specified dataset and load the data based on the mode (train or test).

#### Arguments

- `dataset_name` *str* - The name of the dataset.
- `verbose` *bool* - Whether to print information about the loaded data.
- `log` *bool* - Whether to log-transform the data (log10).
- `normalisation_mode` *str* - The normalization mode, either "disable", "minmax", or "standardise".
- `tolerance` *float, optional* - The tolerance value for log-transformation.
    Values below this will be set to the tolerance value. Pass None to disable.

#### Returns

- `tuple` - Loaded data and timesteps.

#### Raises

- [DatasetError](#dataseterror) - If the dataset or required data is missing or if the data shape is incorrect.

#### Signature

```python
def check_and_load_data(
    dataset_name: str,
    verbose: bool = True,
    log: bool = True,
    normalisation_mode: str = "standardise",
    tolerance: float | None = 1e-20,
): ...
```



## create_dataset

[Show source in data_utils.py:321](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/data_utils.py#L321)

Creates a new dataset in the data directory.

#### Arguments

- `name` *str* - The name of the dataset.
train_data (np.ndarray | torch.Tensor): The training data.
test_data (np.ndarray | torch.Tensor, optional): The test data.
val_data (np.ndarray | torch.Tensor, optional): The validation data.
split tuple(float, float, float), optional): If test_data and val_data are not provided,
    train_data can be split into training, test and validation data.
timesteps (np.ndarray | torch.Tensor, optional): The timesteps array.
- `labels` *list[str], optional* - The labels for the chemicals.

#### Raises

- `FileExistsError` - If the dataset already exists.
- `TypeError` - If the train_data is not a numpy array or torch tensor.
- `ValueError` - If the train_data, test_data, and val_data do not have the correct shape.

#### Signature

```python
def create_dataset(
    name: str,
    train_data: np.ndarray,
    test_data: np.ndarray | None = None,
    val_data: np.ndarray | None = None,
    split: tuple[float, float, float] | None = None,
    timesteps: np.ndarray | None = None,
    labels: list[str] | None = None,
): ...
```



## create_hdf5_dataset

[Show source in data_utils.py:237](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/data_utils.py#L237)

Create an HDF5 file for a dataset with train and test data, and optionally timesteps.
Additionally, store metadata about the dataset.

#### Arguments

- `train_data` *np.ndarray* - The training data array of shape (n_samples, n_timesteps, n_chemicals).
- `test_data` *np.ndarray* - The test data array of shape (n_samples, n_timesteps, n_chemicals).
- `val_data` *np.ndarray* - The validation data array of shape (n_samples, n_timesteps, n_chemicals).
- `dataset_name` *str* - The name of the dataset.
- `data_dir` *str* - The directory to save the dataset in.
- `timesteps` *np.ndarray, optional* - The timesteps array. If None, integer timesteps will be generated.
- `labels` *list[str], optional* - The labels for the chemicals.

#### Signature

```python
def create_hdf5_dataset(
    train_data: np.ndarray,
    test_data: np.ndarray,
    val_data: np.ndarray,
    dataset_name: str,
    data_dir: str = "datasets",
    timesteps: np.ndarray | None = None,
    labels: list[str] | None = None,
): ...
```



## download_data

[Show source in data_utils.py:449](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/data_utils.py#L449)

Download the specified dataset if it is not present

#### Arguments

- `dataset_name` *str* - The name of the dataset.
- `path` *str, optional* - The path to save the dataset. If None, the default data directory is used.

#### Signature

```python
def download_data(dataset_name: str, path: str | None = None): ...
```



## get_data_subset

[Show source in data_utils.py:282](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/data_utils.py#L282)

Get the appropriate data subset based on the mode and metric.

#### Arguments

- `full_train_data` *np.ndarray* - The full training data.
- `full_test_data` *np.ndarray* - The full test data.
- `timesteps` *np.ndarray* - The timesteps.
- `mode` *str* - The benchmark mode (e.g., "accuracy", "interpolation", "extrapolation", "sparse", "UQ").
- `metric` *int* - The specific metric for the mode (e.g., interval, cutoff, factor, batch size).

#### Returns

- `tuple` - The training data, test data, and timesteps subset.

#### Signature

```python
def get_data_subset(full_train_data, full_test_data, timesteps, mode, metric): ...
```



## normalize_data

[Show source in data_utils.py:173](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/utils/data_utils.py#L173)

Normalize the data based on the training data statistics.

#### Arguments

- `train_data` *np.ndarray* - Training data array.
- `test_data` *np.ndarray, optional* - Test data array.
- `val_data` *np.ndarray, optional* - Validation data array.
- `mode` *str* - Normalization mode, either "minmax" or "standardise".

#### Returns

- `tuple` - Normalized training data, test data, and validation data.

#### Signature

```python
def normalize_data(
    train_data: np.ndarray,
    test_data: np.ndarray | None = None,
    val_data: np.ndarray | None = None,
    mode: str = "standardise",
) -> tuple: ...
```