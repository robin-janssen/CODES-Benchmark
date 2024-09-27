# Surrogates

[CODES Index](../../README.md#codes-index) / [Codes](../index.md#codes) / [Surrogates](./index.md#surrogates) / Surrogates

> Auto-generated documentation for [codes.surrogates.surrogates](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/surrogates/surrogates.py) module.

- [Surrogates](#surrogates)
  - [AbstractSurrogateModel](#abstractsurrogatemodel)
    - [AbstractSurrogateModel().denormalize](#abstractsurrogatemodel()denormalize)
    - [AbstractSurrogateModel().fit](#abstractsurrogatemodel()fit)
    - [AbstractSurrogateModel().forward](#abstractsurrogatemodel()forward)
    - [AbstractSurrogateModel().load](#abstractsurrogatemodel()load)
    - [AbstractSurrogateModel().predict](#abstractsurrogatemodel()predict)
    - [AbstractSurrogateModel().prepare_data](#abstractsurrogatemodel()prepare_data)
    - [AbstractSurrogateModel().save](#abstractsurrogatemodel()save)
    - [AbstractSurrogateModel().setup_progress_bar](#abstractsurrogatemodel()setup_progress_bar)

## AbstractSurrogateModel

[Show source in surrogates.py:16](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/surrogates/surrogates.py#L16)

Abstract base class for surrogate models. This class implements the basic
structure of a surrogate model and defines the methods that need to be
implemented by the subclasses for it to be compatible with the benchmarking
framework. For more information, see
https://immi000.github.io/CODES-Benchmark-Docs/documentation.html#add_model.

#### Arguments

- `device` *str, optional* - The device to run the model on. Defaults to None.
- `n_chemicals` *int, optional* - The number of chemicals. Defaults to 29.
- `n_timesteps` *int, optional* - The number of timesteps. Defaults to 100.
- `config` *dict, optional* - The configuration dictionary. Defaults to {}.

#### Attributes

- `train_loss` *float* - The training loss.
- `test_loss` *float* - The test loss.
- `MAE` *float* - The mean absolute error.
- `normalisation` *dict* - The normalisation parameters.
- `train_duration` *float* - The training duration.
- `device` *str* - The device to run the model on.
- `n_chemicals` *int* - The number of chemicals.
- `n_timesteps` *int* - The number of timesteps.
- `L1` *nn.L1Loss* - The L1 loss function.
- `config` *dict* - The configuration dictionary.

#### Methods

- `forward(inputs` - Any) -> tuple[Tensor, Tensor]:
    Forward pass of the model.

prepare_data(
    - `dataset_train` - np.ndarray,
    - `dataset_test` - np.ndarray | None,
    - `dataset_val` - np.ndarray | None,
    - `timesteps` - np.ndarray,
    - `batch_size` - int,
    - `shuffle` - bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    Gets the data loaders for training, testing, and validation.

fit(
    - `train_loader` - DataLoader,
    - `test_loader` - DataLoader,
    - `epochs` - int | None,
    - `position` - int,
    - `description` - str,
) -> None:
    Trains the model on the training data. Sets the train_loss and test_loss attributes.

- `predict(data_loader` - DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    Evaluates the model on the given data loader.

save(
    - `model_name` - str,
    - `subfolder` - str,
    - `training_id` - str,
    - `data_params` - dict,
) -> None:
    Saves the model to disk.

- `load(training_id` - str, surr_name: str, model_identifier: str) -> None:
    Loads a trained surrogate model.

- `setup_progress_bar(epochs` - int, position: int, description: str) -> tqdm:
    Helper function to set up a progress bar for training.

- `denormalize(data` - torch.Tensor) -> torch.Tensor:
    Denormalizes the data back to the original scale.

#### Signature

```python
class AbstractSurrogateModel(ABC, nn.Module):
    def __init__(
        self,
        device: str | None = None,
        n_chemicals: int = 29,
        n_timesteps: int = 100,
        config: dict | None = None,
    ): ...
```

### AbstractSurrogateModel().denormalize

[Show source in surrogates.py:362](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/surrogates/surrogates.py#L362)

Denormalize the data.

#### Arguments

- `data` *np.ndarray* - The data to denormalize.

#### Returns

- `np.ndarray` - The denormalized data.

#### Signature

```python
def denormalize(self, data: torch.Tensor) -> torch.Tensor: ...
```

### AbstractSurrogateModel().fit

[Show source in surrogates.py:147](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/surrogates/surrogates.py#L147)

Perform the training of the model. Sets the train_loss and test_loss attributes.

#### Arguments

- `train_loader` *DataLoader* - The DataLoader object containing the training data.
- `test_loader` *DataLoader* - The DataLoader object containing the testing data.
- `epochs` *int* - The number of epochs to train the model for.
- `position` *int* - The position of the progress bar.
- `description` *str* - The description of the progress bar.

#### Signature

```python
@abstractmethod
def fit(
    self,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    position: int,
    description: str,
) -> None: ...
```

### AbstractSurrogateModel().forward

[Show source in surrogates.py:106](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/surrogates/surrogates.py#L106)

Forward pass of the model.

#### Arguments

- `inputs` *Any* - The input data as recieved from the dataloader.

#### Returns

- `tuple[Tensor,` *Tensor]* - The model predictions and the targets.

#### Signature

```python
@abstractmethod
def forward(self, inputs: Any) -> tuple[Tensor, Tensor]: ...
```

### AbstractSurrogateModel().load

[Show source in surrogates.py:294](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/surrogates/surrogates.py#L294)

Load a trained surrogate model.

#### Arguments

- `training_id` *str* - The training identifier.
- `surr_name` *str* - The name of the surrogate model.
- `model_identifier` *str* - The identifier of the model (e.g., 'main').

#### Returns

None. The model is loaded in place.

#### Signature

```python
def load(
    self,
    training_id: str,
    surr_name: str,
    model_identifier: str,
    model_dir: str | None = None,
) -> None: ...
```

### AbstractSurrogateModel().predict

[Show source in surrogates.py:168](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/surrogates/surrogates.py#L168)

Evaluate the model on the given dataloader.

#### Arguments

- `data_loader` *DataLoader* - The DataLoader object containing the data the
    model is evaluated on.

#### Returns

- `tuple[torch.Tensor,` *torch.Tensor]* - The predictions and targets.

#### Signature

```python
def predict(self, data_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]: ...
```

### AbstractSurrogateModel().prepare_data

[Show source in surrogates.py:119](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/surrogates/surrogates.py#L119)

Prepare the data for training, testing, and validation. This method should
return the DataLoader objects for the training, testing, and validation data.

#### Arguments

- `dataset_train` *np.ndarray* - The training dataset.
- `dataset_test` *np.ndarray* - The testing dataset.
- `dataset_val` *np.ndarray* - The validation dataset.
- `timesteps` *np.ndarray* - The timesteps.
- `batch_size` *int* - The batch size.
- `shuffle` *bool* - Whether to shuffle the data.

#### Returns

tuple[DataLoader, DataLoader, DataLoader]: The DataLoader objects for the
    training, testing, and validation data.

#### Signature

```python
@abstractmethod
def prepare_data(
    self,
    dataset_train: np.ndarray,
    dataset_test: np.ndarray | None,
    dataset_val: np.ndarray | None,
    timesteps: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]: ...
```

### AbstractSurrogateModel().save

[Show source in surrogates.py:219](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/surrogates/surrogates.py#L219)

Save the model to disk.

#### Arguments

- `model_name` *str* - The name of the model.
- `subfolder` *str* - The subfolder to save the model in.
- `training_id` *str* - The training identifier.
- `data_params` *dict* - The data parameters.

#### Signature

```python
def save(
    self, model_name: str, base_dir: str, training_id: str, data_params: dict
) -> None: ...
```

### AbstractSurrogateModel().setup_progress_bar

[Show source in surrogates.py:337](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/surrogates/surrogates.py#L337)

Helper function to set up a progress bar for training.

#### Arguments

- `epochs` *int* - The number of epochs.
- `position` *int* - The position of the progress bar.
- `description` *str* - The description of the progress bar.

#### Returns

- `tqdm` - The progress bar.

#### Signature

```python
def setup_progress_bar(self, epochs: int, position: int, description: str): ...
```