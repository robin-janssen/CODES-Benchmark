# Data Plots

[Codes-benchmark Index](../../README.md#codes-benchmark-index) / `datasets` / [Data Analysis](./index.md#data-analysis) / Data Plots

> Auto-generated documentation for [datasets.data_analysis.data_plots](https://github.com/robin-janssen/CODES-Benchmark/blob/main/datasets/data_analysis/data_plots.py) module.

- [Data Plots](#data-plots)
  - [plot_example_trajectories](#plot_example_trajectories)
  - [plot_example_trajectories_paper](#plot_example_trajectories_paper)

## plot_example_trajectories

[Show source in data_plots.py:11](https://github.com/robin-janssen/CODES-Benchmark/blob/main/datasets/data_analysis/data_plots.py#L11)

Plot example trajectories for the dataset.

#### Arguments

- `dataset_name` *str* - The name of the dataset.
- `data` *np.ndarray* - The data to plot.
- `timesteps` *np.ndarray* - Timesteps array.
- `num_chemicals` *int, optional* - Number of chemicals to plot. Default is 10.
- `labels` *list, optional* - List of labels for the chemicals.
- `save` *bool, optional* - Whether to save the plot as a file.

#### Signature

```python
def plot_example_trajectories(
    dataset_name: str,
    data: np.ndarray,
    timesteps: np.ndarray,
    num_chemicals: int = 10,
    labels: list[str] | None = None,
    save: bool = False,
    sample_idx: int = 0,
    log: bool = False,
) -> None: ...
```



## plot_example_trajectories_paper

[Show source in data_plots.py:88](https://github.com/robin-janssen/CODES-Benchmark/blob/main/datasets/data_analysis/data_plots.py#L88)

Plot example trajectories for the dataset with two subplots, one showing 15 chemicals and another showing the remaining.

#### Arguments

- `dataset_name` *str* - The name of the dataset.
- `data` *np.ndarray* - The data to plot.
- `timesteps` *np.ndarray* - Timesteps array.
- `save` *bool, optional* - Whether to save the plot as a file.
- `sample_idx` *int, optional* - Index of the sample to plot.
- `labels` *list, optional* - List of labels for the chemicals.

#### Signature

```python
def plot_example_trajectories_paper(
    dataset_name: str,
    data: np.ndarray,
    timesteps: np.ndarray,
    save: bool = False,
    sample_idx: int = 0,
    labels: list[str] | None = None,
) -> None: ...
```