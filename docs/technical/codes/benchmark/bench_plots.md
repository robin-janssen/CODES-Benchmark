# Bench Plots

[Codes-benchmark Index](../../README.md#codes-benchmark-index) / [Codes](../index.md#codes) / [Benchmark](./index.md#benchmark) / Bench Plots

> Auto-generated documentation for [codes.benchmark.bench_plots](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py) module.

- [Bench Plots](#bench-plots)
  - [get_custom_palette](#get_custom_palette)
  - [inference_time_bar_plot](#inference_time_bar_plot)
  - [int_ext_sparse](#int_ext_sparse)
  - [plot_MAE_comparison](#plot_mae_comparison)
  - [plot_MAE_comparison_train_duration](#plot_mae_comparison_train_duration)
  - [plot_average_errors_over_time](#plot_average_errors_over_time)
  - [plot_average_uncertainty_over_time](#plot_average_uncertainty_over_time)
  - [plot_comparative_dynamic_correlation_heatmaps](#plot_comparative_dynamic_correlation_heatmaps)
  - [plot_comparative_error_correlation_heatmaps](#plot_comparative_error_correlation_heatmaps)
  - [plot_dynamic_correlation](#plot_dynamic_correlation)
  - [plot_dynamic_correlation_heatmap](#plot_dynamic_correlation_heatmap)
  - [plot_error_correlation_heatmap](#plot_error_correlation_heatmap)
  - [plot_error_distribution_comparative](#plot_error_distribution_comparative)
  - [plot_error_distribution_per_chemical](#plot_error_distribution_per_chemical)
  - [plot_example_predictions_with_uncertainty](#plot_example_predictions_with_uncertainty)
  - [plot_generalization_error_comparison](#plot_generalization_error_comparison)
  - [plot_generalization_errors](#plot_generalization_errors)
  - [plot_loss_comparison](#plot_loss_comparison)
  - [plot_losses](#plot_losses)
  - [plot_relative_errors](#plot_relative_errors)
  - [plot_relative_errors_over_time](#plot_relative_errors_over_time)
  - [plot_surr_losses](#plot_surr_losses)
  - [plot_uncertainty_over_time_comparison](#plot_uncertainty_over_time_comparison)
  - [plot_uncertainty_vs_errors](#plot_uncertainty_vs_errors)
  - [rel_errors_and_uq](#rel_errors_and_uq)
  - [save_plot](#save_plot)
  - [save_plot_counter](#save_plot_counter)

## get_custom_palette

[Show source in bench_plots.py:1645](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L1645)

Returns a list of colors sampled from a custom color palette.

#### Arguments

- `num_colors` *int* - The number of colors needed.

#### Returns

- `list` - A list of RGBA color tuples.

#### Signature

```python
def get_custom_palette(num_colors): ...
```



## inference_time_bar_plot

[Show source in bench_plots.py:1065](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L1065)

Plot the mean inference time with standard deviation for different surrogate models.

#### Arguments

- `surrogates` *List[str]* - List of surrogate model names.
- `means` *List[float]* - List of mean inference times for each surrogate model.
- `stds` *List[float]* - List of standard deviation of inference times for each surrogate model.
- `config` *dict* - Configuration dictionary.
- `save` *bool, optional* - Whether to save the plot. Defaults to True.

#### Returns

None

#### Signature

```python
def inference_time_bar_plot(
    surrogates: list[str],
    means: list[float],
    stds: list[float],
    config: dict,
    save: bool = True,
) -> None: ...
```



## int_ext_sparse

[Show source in bench_plots.py:1691](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L1691)

Function to make one comparative plot of the interpolation, extrapolation, and sparse training errors.

#### Arguments

- `all_metrics` *dict* - dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.

#### Returns

None

#### Signature

```python
def int_ext_sparse(all_metrics: dict, config: dict) -> None: ...
```



## plot_MAE_comparison

[Show source in bench_plots.py:873](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L873)

Plot the MAE for different surrogate models.

#### Arguments

- `MAE` *tuple* - Tuple of accuracy arrays for each surrogate model.
- `labels` *tuple* - Tuple of labels for each surrogate model.
- `config` *dict* - Configuration dictionary.
- `save` *bool* - Whether to save the plot.

#### Signature

```python
def plot_MAE_comparison(
    MAEs: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    config: dict,
    save: bool = True,
) -> None: ...
```



## plot_MAE_comparison_train_duration

[Show source in bench_plots.py:910](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L910)

Plot the MAE for different surrogate models.

#### Arguments

- `MAE` *tuple* - Tuple of accuracy arrays for each surrogate model.
- `labels` *tuple* - Tuple of labels for each surrogate model.
- `config` *dict* - Configuration dictionary.
- `save` *bool* - Whether to save the plot.

#### Signature

```python
def plot_MAE_comparison_train_duration(
    MAEs: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    train_durations: tuple[float, ...],
    config: dict,
    save: bool = True,
) -> None: ...
```



## plot_average_errors_over_time

[Show source in bench_plots.py:258](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L258)

Plot the errors over time for different modes (interpolation, extrapolation, sparse, batchsize).

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `errors` *np.ndarray* - Errors array of shape [N_metrics, n_timesteps].
- `metrics` *np.ndarray* - Metrics array of shape [N_metrics].
- `timesteps` *np.ndarray* - Timesteps array.
- `mode` *str* - The mode of evaluation ('interpolation', 'extrapolation', 'sparse', 'batchsize').
- `save` *bool, optional* - Whether to save the plot as a file.

#### Signature

```python
def plot_average_errors_over_time(
    surr_name: str,
    conf: dict,
    errors: np.ndarray,
    metrics: np.ndarray,
    timesteps: np.ndarray,
    mode: str,
    save: bool = False,
) -> None: ...
```



## plot_average_uncertainty_over_time

[Show source in bench_plots.py:455](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L455)

Plot the average uncertainty over time.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `errors_time` *np.ndarray* - Prediction errors over time.
- `preds_std` *np.ndarray* - Standard deviation over time of predictions from the ensemble of models.
- `timesteps` *np.ndarray* - Timesteps array.
- `save` *bool, optional* - Whether to save the plot as a file.

#### Signature

```python
def plot_average_uncertainty_over_time(
    surr_name: str,
    conf: dict,
    errors_time: np.ndarray,
    preds_std: np.ndarray,
    timesteps: np.ndarray,
    save: bool = False,
) -> None: ...
```



## plot_comparative_dynamic_correlation_heatmaps

[Show source in bench_plots.py:1550](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L1550)

Plot comparative heatmaps of correlation between gradient and prediction errors
for multiple surrogate models.

#### Arguments

gradients (dict[str, np.ndarray]): Dictionary of gradients from the ensemble of models.
errors (dict[str, np.ndarray]): Dictionary of prediction errors.
avg_correlations (dict[str, float]): Dictionary of average correlations between gradients and prediction errors.
max_grad (dict[str, float]): Dictionary of maximum gradient values for axis scaling across models.
max_err (dict[str, float]): Dictionary of maximum error values for axis scaling across models.
max_count (dict[str, float]): Dictionary of maximum count values for heatmap normalization across models.
- `config` *dict* - Configuration dictionary.
- `save` *bool, optional* - Whether to save the plot. Defaults to True.

#### Returns

None

#### Signature

```python
def plot_comparative_dynamic_correlation_heatmaps(
    gradients: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    avg_correlations: dict[str, float],
    max_grad: dict[str, float],
    max_err: dict[str, float],
    max_count: dict[str, float],
    config: dict,
    save: bool = True,
) -> None: ...
```



## plot_comparative_error_correlation_heatmaps

[Show source in bench_plots.py:1453](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L1453)

Plot comparative heatmaps of correlation between predictive uncertainty and prediction errors
for multiple surrogate models.

#### Arguments

preds_std (dict[str, np.ndarray]): Dictionary of standard deviation of predictions from the ensemble of models.
errors (dict[str, np.ndarray]): Dictionary of prediction errors.
avg_correlations (dict[str, float]): Dictionary of average correlations between gradients and prediction errors.
axis_max (dict[str, float]): Dictionary of maximum values for axis scaling across models.
max_count (dict[str, float]): Dictionary of maximum count values for heatmap normalization across models.
- `config` *dict* - Configuration dictionary.
- `save` *bool, optional* - Whether to save the plot. Defaults to True.

#### Returns

None

#### Signature

```python
def plot_comparative_error_correlation_heatmaps(
    preds_std: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    avg_correlations: dict[str, float],
    axis_max: dict[str, float],
    max_count: dict[str, float],
    config: dict,
    save: bool = True,
) -> None: ...
```



## plot_dynamic_correlation

[Show source in bench_plots.py:163](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L163)

Plot the correlation between the gradients of the data and the prediction errors.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `gradients` *np.ndarray* - The gradients of the data.
- `errors` *np.ndarray* - The prediction errors.
- `save` *bool* - Whether to save the plot.

#### Signature

```python
def plot_dynamic_correlation(
    surr_name: str,
    conf: dict,
    gradients: np.ndarray,
    errors: np.ndarray,
    save: bool = False,
): ...
```



## plot_dynamic_correlation_heatmap

[Show source in bench_plots.py:1269](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L1269)

Plot the correlation between predictive uncertainty and prediction errors using a heatmap.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `preds_std` *np.ndarray* - Standard deviation of predictions from the ensemble of models.
- `errors` *np.ndarray* - Prediction errors.
- `average_correlation` *float* - The average correlation between gradients and prediction errors (pearson correlation).
- `save` *bool, optional* - Whether to save the plot as a file.
- `threshold_factor` *float, optional* - Fraction of max value below which cells are set to 0. Default is 5e-5.
- `cutoff_percent` *float, optional* - The percentage of total counts to include in the heatmap. Default is 0.95.

#### Signature

```python
def plot_dynamic_correlation_heatmap(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    errors: np.ndarray,
    average_correlation: float,
    save: bool = False,
    threshold_factor: float = 0.0001,
    xcut_percent: float = 0.003,
) -> None: ...
```



## plot_error_correlation_heatmap

[Show source in bench_plots.py:1181](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L1181)

Plot the correlation between predictive uncertainty and prediction errors using a heatmap.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `preds_std` *np.ndarray* - Standard deviation of predictions from the ensemble of models.
- `errors` *np.ndarray* - Prediction errors.
- `average_correlation` *float* - The average correlation between gradients and prediction errors (pearson correlation).
- `save` *bool, optional* - Whether to save the plot as a file.
- `threshold_factor` *float, optional* - Fraction of max value below which cells are set to 0. Default is 0.001.

#### Signature

```python
def plot_error_correlation_heatmap(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    errors: np.ndarray,
    average_correlation: float,
    save: bool = False,
    threshold_factor: float = 0.01,
) -> None: ...
```



## plot_error_distribution_comparative

[Show source in bench_plots.py:1354](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L1354)

Plot the comparative distribution of errors for each surrogate model as a smoothed histogram plot.

#### Arguments

- `conf` *dict* - The configuration dictionary.
- `errors` *dict* - Dictionary containing numpy arrays of shape [num_samples, num_timesteps, num_chemicals] for each model.
- `save` *bool, optional* - Whether to save the plot as a file.

#### Signature

```python
def plot_error_distribution_comparative(
    errors: dict[str, np.ndarray], conf: dict, save: bool = True
) -> None: ...
```



## plot_error_distribution_per_chemical

[Show source in bench_plots.py:655](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L655)

Plot the distribution of errors for each chemical as a smoothed histogram plot.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `errors` *np.ndarray* - Errors array of shape [num_samples, num_timesteps, num_chemicals].
- `chemical_names` *list, optional* - List of chemical names for labeling the lines.
- `num_chemicals` *int, optional* - Number of chemicals to plot. Default is 10.
- `save` *bool, optional* - Whether to save the plot as a file.

#### Signature

```python
def plot_error_distribution_per_chemical(
    surr_name: str,
    conf: dict,
    errors: np.ndarray,
    chemical_names: list[str] | None = None,
    num_chemicals: int = 10,
    save: bool = True,
) -> None: ...
```



## plot_example_predictions_with_uncertainty

[Show source in bench_plots.py:331](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L331)

Plot example predictions with uncertainty.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `preds_mean` *np.ndarray* - Mean predictions from the ensemble of models.
- `preds_std` *np.ndarray* - Standard deviation of predictions from the ensemble of models.
- `targets` *np.ndarray* - True targets.
- `timesteps` *np.ndarray* - Timesteps array.
- `example_idx` *int, optional* - Index of the example to plot. Default is 0.
- `num_chemicals` *int, optional* - Number of chemicals to plot. Default is 100.
- `labels` *list, optional* - List of labels for the chemicals.
- `save` *bool, optional* - Whether to save the plot as a file.

#### Signature

```python
def plot_example_predictions_with_uncertainty(
    surr_name: str,
    conf: dict,
    preds_mean: np.ndarray,
    preds_std: np.ndarray,
    targets: np.ndarray,
    timesteps: np.ndarray,
    example_idx: int = 0,
    num_chemicals: int = 100,
    labels: list[str] | None = None,
    save: bool = False,
) -> None: ...
```



## plot_generalization_error_comparison

[Show source in bench_plots.py:1123](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L1123)

Plot the generalization errors of different surrogate models.

#### Arguments

- `surrogates` *list* - List of surrogate model names.
- `metrics_list` *list[np.array]* - List of numpy arrays containing the metrics for each surrogate model.
- `model_errors_list` *list[np.array]* - List of numpy arrays containing the errors for each surrogate model.
- `xlabel` *str* - Label for the x-axis.
- `filename` *str* - Filename to save the plot.
- `config` *dict* - Configuration dictionary.
- `save` *bool* - Whether to save the plot.
- `xlog` *bool* - Whether to use a log scale for the x-axis.

#### Returns

None

#### Signature

```python
def plot_generalization_error_comparison(
    surrogates: list[str],
    metrics_list: list[np.array],
    model_errors_list: list[np.array],
    xlabel: str,
    filename: str,
    config: dict,
    save: bool = True,
    xlog: bool = False,
) -> None: ...
```



## plot_generalization_errors

[Show source in bench_plots.py:198](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L198)

Plot the generalization errors of a model for various metrics.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `metrics` *np.ndarray* - The metrics (e.g., intervals, cutoffs, batch sizes, number of training samples).
- `model_errors` *np.ndarray* - The model errors.
- `mode` *str* - The mode of generalization ("interpolation", "extrapolation", "sparse", "batchsize").
- `save` *bool* - Whether to save the plot.

#### Returns

None

#### Signature

```python
def plot_generalization_errors(
    surr_name: str,
    conf: dict,
    metrics: np.ndarray,
    model_errors: np.ndarray,
    mode: str,
    save: bool = False,
) -> None: ...
```



## plot_loss_comparison

[Show source in bench_plots.py:828](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L828)

Plot the training and test losses for different surrogate models.

#### Arguments

- `train_losses` *tuple* - Tuple of training loss arrays for each surrogate model.
- `test_losses` *tuple* - Tuple of test loss arrays for each surrogate model.
- `labels` *tuple* - Tuple of labels for each surrogate model.
- `config` *dict* - Configuration dictionary.
- `save` *bool* - Whether to save the plot.

#### Returns

None

#### Signature

```python
def plot_loss_comparison(
    train_losses: tuple[np.ndarray, ...],
    test_losses: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    config: dict,
    save: bool = True,
) -> None: ...
```



## plot_losses

[Show source in bench_plots.py:775](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L775)

Plot the loss trajectories for the training of multiple models.

#### Arguments

- `loss_histories` - List of loss history arrays.
- `labels` - List of labels for each loss history.
- `title` - Title of the plot.
- `save` - Whether to save the plot as an image file.
- `conf` - The configuration dictionary.
- `surr_name` - The name of the surrogate model.
- `mode` - The mode of the training.

#### Signature

```python
def plot_losses(
    loss_histories: tuple[np.array, ...],
    labels: tuple[str, ...],
    title: str = "Losses",
    save: bool = False,
    conf: Optional[dict] = None,
    surr_name: Optional[str] = None,
    mode: str = "main",
) -> None: ...
```



## plot_relative_errors

[Show source in bench_plots.py:949](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L949)

Plot the relative errors over time for different surrogate models.

#### Arguments

- `mean_errors` *dict* - dictionary containing the mean relative errors for each surrogate model.
- `median_errors` *dict* - dictionary containing the median relative errors for each surrogate model.
- `timesteps` *np.ndarray* - Array of timesteps.
- `config` *dict* - Configuration dictionary.
- `save` *bool* - Whether to save the plot.

#### Returns

None

#### Signature

```python
def plot_relative_errors(
    mean_errors: dict[str, np.ndarray],
    median_errors: dict[str, np.ndarray],
    timesteps: np.ndarray,
    config: dict,
    save: bool = True,
) -> None: ...
```



## plot_relative_errors_over_time

[Show source in bench_plots.py:86](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L86)

Plot the mean and median relative errors over time with shaded regions for
the 50th, 90th, and 99th percentiles.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `relative_errors` *np.ndarray* - The relative errors of the model.
- `title` *str* - The title of the plot.
- `save` *bool* - Whether to save the plot.

#### Signature

```python
def plot_relative_errors_over_time(
    surr_name: str,
    conf: dict,
    relative_errors: np.ndarray,
    title: str,
    save: bool = False,
) -> None: ...
```



## plot_surr_losses

[Show source in bench_plots.py:525](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L525)

Plot the training and test losses for the surrogate model.

#### Arguments

- `model` - Instance of the surrogate model class.
- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `timesteps` *np.ndarray* - The timesteps array.

#### Signature

```python
def plot_surr_losses(
    model, surr_name: str, conf: dict, timesteps: np.ndarray
) -> None: ...
```



## plot_uncertainty_over_time_comparison

[Show source in bench_plots.py:1007](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L1007)

Plot the uncertainty over time for different surrogate models.

#### Arguments

- `uncertainties` *dict* - Dictionary containing the uncertainties for each surrogate model.
- `absolute_errors` *dict* - Dictionary containing the absolute errors for each surrogate model.
- `timesteps` *np.ndarray* - Array of timesteps.
- `config` *dict* - Configuration dictionary.
- `save` *bool* - Whether to save the plot.

#### Returns

None

#### Signature

```python
def plot_uncertainty_over_time_comparison(
    uncertainties: dict[str, np.ndarray],
    absolute_errors: dict[str, np.ndarray],
    timesteps: np.ndarray,
    config: dict,
    save: bool = True,
) -> None: ...
```



## plot_uncertainty_vs_errors

[Show source in bench_plots.py:493](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L493)

Plot the correlation between predictive uncertainty and prediction errors.

#### Arguments

- `surr_name` *str* - The name of the surrogate model.
- `conf` *dict* - The configuration dictionary.
- `preds_std` *np.ndarray* - Standard deviation of predictions from the ensemble of models.
- `errors` *np.ndarray* - Prediction errors.
- `save` *bool, optional* - Whether to save the plot as a file.

#### Signature

```python
def plot_uncertainty_vs_errors(
    surr_name: str,
    conf: dict,
    preds_std: np.ndarray,
    errors: np.ndarray,
    save: bool = False,
) -> None: ...
```



## rel_errors_and_uq

[Show source in bench_plots.py:1830](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L1830)

Create a figure with two subplots: relative errors over time and uncertainty over time for different surrogate models.

#### Arguments

- `metrics` *dict* - Dictionary containing the benchmark metrics for each surrogate model.
- `config` *dict* - Configuration dictionary.
- `save` *bool* - Whether to save the plot.

#### Returns

None

#### Signature

```python
def rel_errors_and_uq(
    metrics: dict[str, dict], config: dict, save: bool = True
) -> None: ...
```



## save_plot

[Show source in bench_plots.py:15](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L15)

Save the plot to a file, creating necessary directories if they don't exist.

#### Arguments

- `plt` *matplotlib.pyplot* - The plot object to save.
- `filename` *str* - The desired filename for the plot.
- `conf` *dict* - The configuration dictionary.
- `surr_name` *str* - The name of the surrogate model.
- `dpi` *int* - The resolution of the saved plot.
- `base_dir` *str, optional* - The base directory where plots will be saved. Default is "plots".
- `increase_count` *bool, optional* - Whether to increment the filename count if a file already exists. Default is True.

#### Raises

- `ValueError` - If the configuration dictionary does not contain the required keys.

#### Signature

```python
def save_plot(
    plt,
    filename: str,
    conf: dict,
    surr_name: str = "",
    dpi: int = 300,
    base_dir: str = "plots",
    increase_count: bool = False,
) -> None: ...
```



## save_plot_counter

[Show source in bench_plots.py:54](https://github.com/robin-janssen/CODES-Benchmark/blob/main/codes/benchmark/bench_plots.py#L54)

Save a plot with an incremented filename if a file with the same name already exists.

#### Arguments

- `filename` *str* - The desired filename for the plot.
- `directory` *str* - The directory to save the plot in.
- `increase_count` *bool, optional* - Whether to increment the filename count if a file already exists. Default is True.

#### Returns

- `str` - The full path to the saved plot.

#### Signature

```python
def save_plot_counter(
    filename: str, directory: str, increase_count: bool = True
) -> str: ...
```