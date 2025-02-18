import time
from contextlib import redirect_stdout
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr
from tabulate import tabulate
from torch.utils.data import DataLoader

from codes.utils import check_and_load_data

from .bench_plots import (
    inference_time_bar_plot,
    int_ext_sparse,
    plot_average_errors_over_time,
    plot_average_uncertainty_over_time,
    plot_comparative_dynamic_correlation_heatmaps,
    plot_comparative_error_correlation_heatmaps,
    plot_dynamic_correlation_heatmap,
    plot_error_correlation_heatmap,
    plot_error_distribution_comparative,
    plot_error_distribution_per_chemical,
    plot_example_predictions_with_uncertainty,
    plot_generalization_error_comparison,
    plot_generalization_errors,
    plot_loss_comparison,
    plot_MAE_comparison_train_duration,
    plot_relative_errors,
    plot_relative_errors_over_time,
    plot_surr_losses,
    plot_uncertainty_over_time_comparison,
    rel_errors_and_uq,
)
from .bench_utils import (
    count_trainable_parameters,
    format_seconds,
    format_time,
    get_model_config,
    get_surrogate,
    make_comparison_csv,
    measure_memory_footprint,
    write_metrics_to_yaml,
)


def run_benchmark(surr_name: str, surrogate_class, conf: dict) -> dict[str, Any]:
    """
    Run benchmarks for a given surrogate model.

    Args:
        surr_name (str): The name of the surrogate model to benchmark.
        surrogate_class: The class of the surrogate model.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing all relevant metrics for the given model.
    """
    # Instantiate the model
    device = conf["devices"]
    device = device[0] if isinstance(device, list) else device

    # Determine the batch size
    surr_idx = conf["surrogates"].index(surr_name)
    if isinstance(conf["batch_size"], list):
        if len(conf["batch_size"]) != len(conf["surrogates"]):
            raise ValueError(
                "The number of provided batch sizes must match the number of surrogate models."
            )
        else:
            batch_size = conf["batch_size"][surr_idx]
    else:
        batch_size = conf["batch_size"]

    train_data, test_data, val_data, timesteps, n_train_samples, _, labels = (
        check_and_load_data(
            conf["dataset"]["name"],
            verbose=False,
            log=conf["dataset"]["log10_transform"],
            normalisation_mode=conf["dataset"]["normalise"],
        )
    )
    model_config = get_model_config(surr_name, conf)
    n_timesteps = train_data.shape[1]
    n_chemicals = train_data.shape[2]
    n_test_samples = n_timesteps * val_data.shape[0]
    model = surrogate_class(device, n_chemicals, n_timesteps, model_config)

    # Placeholder for metrics
    metrics = {}
    metrics["timesteps"] = timesteps

    # Create dataloader for the validation data
    _, _, val_loader = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=val_data,
        timesteps=timesteps,
        batch_size=batch_size,
        shuffle=False,
    )

    # Plot training losses
    if conf["losses"]:
        print("Loss plots...")
        plot_surr_losses(model, surr_name, conf, timesteps)

    # Accuracy benchmark
    print("Running accuracy benchmark...")
    metrics["accuracy"] = evaluate_accuracy(model, surr_name, val_loader, conf, labels)

    # Gradients benchmark
    if conf["gradients"]:
        print("Running gradients benchmark...")
        # For this benchmark, we can also use the main model
        metrics["gradients"] = evaluate_dynamic_accuracy(
            model, surr_name, val_loader, conf
        )

    # Timing benchmark
    if conf["timing"]:
        print("Running timing benchmark...")
        metrics["timing"] = time_inference(
            model, surr_name, val_loader, conf, n_test_samples
        )

    # Compute (resources) benchmark
    if conf["compute"]:
        print("Running compute benchmark...")
        metrics["compute"] = evaluate_compute(model, surr_name, val_loader, conf)

    # Interpolation benchmark
    if conf["interpolation"]["enabled"]:
        print("Running interpolation benchmark...")
        metrics["interpolation"] = evaluate_interpolation(
            model, surr_name, val_loader, timesteps, conf
        )

    # Extrapolation benchmark
    if conf["extrapolation"]["enabled"]:
        print("Running extrapolation benchmark...")
        metrics["extrapolation"] = evaluate_extrapolation(
            model, surr_name, val_loader, timesteps, conf
        )

    # Sparse data benchmark
    if conf["sparse"]["enabled"]:
        print("Running sparse benchmark...")
        metrics["sparse"] = evaluate_sparse(
            model, surr_name, val_loader, timesteps, n_train_samples, conf
        )

    # Batch size benchmark
    if conf["batch_scaling"]["enabled"]:
        print("Running batch size benchmark...")
        metrics["batch_size"] = evaluate_batchsize(
            model, surr_name, val_loader, timesteps, conf
        )

    # Uncertainty Quantification (UQ) benchmark
    if conf["uncertainty"]["enabled"]:
        print("Running UQ benchmark...")
        metrics["UQ"] = evaluate_UQ(
            model, surr_name, val_loader, timesteps, conf, labels
        )

    # Write metrics to yaml
    write_metrics_to_yaml(surr_name, conf, metrics)

    return metrics


def evaluate_accuracy(
    model,
    surr_name: str,
    test_loader: DataLoader,
    conf: dict,
    labels: list | None = None,
) -> dict[str, Any]:
    """
    Evaluate the accuracy of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        conf (dict): The configuration dictionary.
        labels (list, optional): The labels for the chemical species.

    Returns:
        dict: A dictionary containing accuracy metrics.
    """
    training_id = conf["training_id"]

    # Load the model
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")
    train_time = model.train_duration
    num_chemicals = model.n_chemicals
    model_index = conf["surrogates"].index(surr_name)
    n_epochs = conf["epochs"][model_index]

    # Use the model's predict method
    criterion = torch.nn.MSELoss()
    preds, targets = model.predict(data_loader=test_loader)
    mean_squared_error = criterion(preds, targets).item()  # / torch.numel(preds)
    preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()

    # Calculate relative errors
    absolute_errors = np.abs(preds - targets)
    mean_absolute_error = np.mean(absolute_errors)
    relative_errors = np.abs(absolute_errors / targets)

    # Plot relative errors over time
    plot_relative_errors_over_time(
        surr_name,
        conf,
        relative_errors,
        title=f"Relative Errors over Time for {surr_name}",
        save=True,
    )

    plot_error_distribution_per_chemical(
        surr_name,
        conf,
        relative_errors,
        chemical_names=labels,
        num_chemicals=num_chemicals,
        save=True,
    )

    # Store metrics
    accuracy_metrics = {
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_relative_error": np.mean(relative_errors),
        "median_relative_error": np.median(relative_errors),
        "max_relative_error": np.max(relative_errors),
        "min_relative_error": np.min(relative_errors),
        "absolute_errors": absolute_errors,
        "relative_errors": relative_errors,
        "main_model_training_time": train_time,
        "main_model_epochs": n_epochs,
    }

    return accuracy_metrics


def evaluate_dynamic_accuracy(
    model,
    surr_name: str,
    test_loader: DataLoader,
    conf: dict,
    species_names: list = None,
) -> dict:
    """
    Evaluate the gradients of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing gradients metrics.
    """
    training_id = conf["training_id"]

    # Load the model
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")

    # Obtain predictions and targets
    preds, targets = model.predict(data_loader=test_loader)
    preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()

    # Calculate gradients of the target data w.r.t time
    gradients = np.gradient(targets, axis=1)
    # Take absolute value and normalize gradients
    gradients = np.abs(gradients) / np.abs(gradients).max()

    # Calculate absolute prediction errors
    prediction_errors = np.abs(preds - targets)
    # Normalize prediction errors
    # prediction_errors = prediction_errors / prediction_errors.max()

    # Calculate correlations
    species_correlations = []
    for i in range(targets.shape[2]):
        gradient_species = gradients[:, :, i].flatten()
        error_species = prediction_errors[:, :, i].flatten()
        correlation, _ = pearsonr(gradient_species, error_species)
        species_correlations.append(correlation)

    # Average correlation over all species
    avg_gradient = gradients.mean(axis=2).flatten()
    avg_error = prediction_errors.mean(axis=2).flatten()
    avg_correlation, _ = pearsonr(avg_gradient, avg_error)

    # Plot correlation for averaged species
    # plot_dynamic_correlation(surr_name, conf, avg_gradient, avg_error, save=True)
    max_count, max_grad, max_err = plot_dynamic_correlation_heatmap(
        surr_name, conf, gradients, prediction_errors, avg_correlation, save=True
    )

    # Ensure species names are provided
    species_names = (
        species_names
        if species_names is not None
        else [f"quantity_{i}" for i in range(targets.shape[2])]
    )
    species_correlations = dict(zip(species_names, species_correlations))

    # Store metrics
    gradients_metrics = {
        "gradients": gradients,
        "species_correlations": species_correlations,
        "avg_correlation": avg_correlation,
        "max_counts": max_count,
        "max_gradient": max_grad,
        "max_error": max_err,
    }

    return gradients_metrics


def time_inference(
    model,
    surr_name: str,
    test_loader: DataLoader,
    conf: dict,
    n_test_samples: int,
    n_runs: int = 5,
) -> dict[str, Any]:
    """
    Time the inference of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        timesteps (np.ndarray): The timesteps array.
        conf (dict): The configuration dictionary.
        n_test_samples (int): The number of test samples.
        n_runs (int, optional): Number of times to run the inference for timing.

    Returns:
        dict: A dictionary containing timing metrics.
    """
    training_id = conf["training_id"]
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")

    # Run inference multiple times and record the durations
    inference_times = []
    for _ in range(n_runs):
        # _, _ = model.predict(data_loader=test_loader)
        total_time = 0
        with torch.inference_mode():
            for inputs in test_loader:
                start_time = time.perf_counter()
                _, _ = model.forward(inputs)
                end_time = time.perf_counter()
                total_time += end_time - start_time
        # total_time /= n_test_samples
        inference_times.append(total_time)

    # Calculate metrics
    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    # Store metrics
    timing_metrics = {
        "mean_inference_time_per_run": mean_inference_time,
        "std_inference_time_per_run": std_inference_time,
        "num_predictions": n_test_samples,
        "mean_inference_time_per_prediction": mean_inference_time / n_test_samples,
        "std_inference_time_per_prediction": std_inference_time / n_test_samples,
    }

    return timing_metrics


def evaluate_compute(
    model, surr_name: str, test_loader: DataLoader, conf: dict
) -> dict[str, Any]:
    """
    Evaluate the computational resource requirements of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing model complexity metrics.
    """
    training_id = conf["training_id"]
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")

    # Get a sample input tensor from the test_loader
    inputs = next(iter(test_loader))
    # Measure the memory footprint during forward and backward pass
    memory_footprint, model = measure_memory_footprint(model, inputs)

    # Count the number of trainable parameters
    num_params = count_trainable_parameters(model)

    # Store complexity metrics
    complexity_metrics = {
        "num_trainable_parameters": num_params,
        "memory_footprint": memory_footprint,
    }

    return complexity_metrics


def evaluate_interpolation(
    model, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: dict
) -> dict[str, Any]:
    """
    Evaluate the interpolation performance of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        timesteps (np.ndarray): The timesteps array.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing interpolation metrics.
    """
    training_id = conf["training_id"]
    intervals = conf["interpolation"]["intervals"]
    intervals = np.sort(np.array(intervals, dtype=int))
    intervals = intervals[intervals > 1]
    intervals = np.insert(intervals, 0, 1)
    interpolation_metrics = {}
    errors = np.zeros((len(intervals), len(timesteps)))

    # Criterion for prediction loss
    criterion = torch.nn.MSELoss()

    # Evaluate models for each interval
    for interval in intervals:
        # Ensure that the main model is loaded for interval 1
        model_id = (
            f"{surr_name.lower()}_main"
            if interval == 1
            else f"{surr_name.lower()}_interpolation_{interval}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)
        preds, targets = model.predict(data_loader=test_loader)
        mean_squared_error = criterion(preds, targets).item()  # / torch.numel(preds)
        interpolation_metrics[f"interval {interval}"] = {"MSE": mean_squared_error}

        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        mean_absolute_errors = np.mean(np.abs(preds - targets), axis=(0, 2))
        errors[intervals == interval] = mean_absolute_errors
        mean_absolute_error = np.mean(mean_absolute_errors)
        interpolation_metrics[f"interval {interval}"]["MAE"] = mean_absolute_error

    # Extract metrics and errors for plotting
    model_errors = np.array(
        # [metric["MSE"] for metric in interpolation_metrics.values()]
        [metric["MAE"] for metric in interpolation_metrics.values()]
    )
    interpolation_metrics["model_errors"] = model_errors
    interpolation_metrics["intervals"] = intervals

    # Plot interpolation errors
    plot_generalization_errors(
        surr_name, conf, intervals, model_errors, mode="interpolation", save=True
    )
    plot_average_errors_over_time(
        surr_name,
        conf,
        errors,
        intervals,
        timesteps,
        mode="interpolation",
        save=True,
    )

    return interpolation_metrics


def evaluate_extrapolation(
    model, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: dict
) -> dict[str, Any]:
    """
    Evaluate the extrapolation performance of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        timesteps (np.ndarray): The timesteps array.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing extrapolation metrics.
    """
    training_id = conf["training_id"]
    cutoffs = conf["extrapolation"]["cutoffs"]
    cutoffs = np.sort(np.array(cutoffs, dtype=int))
    max_cut = len(timesteps)
    cutoffs = cutoffs[cutoffs < max_cut]
    cutoffs = np.append(cutoffs, max_cut)
    extrapolation_metrics = {}
    errors = np.zeros((len(cutoffs), len(timesteps)))

    # Criterion for prediction loss
    criterion = torch.nn.MSELoss()

    # Evaluate models for each cutoff
    for cutoff in cutoffs:
        # Ensure that the main model is loaded for the last cutoff
        model_id = (
            f"{surr_name.lower()}_main"
            if cutoff == max_cut
            else f"{surr_name.lower()}_extrapolation_{cutoff}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)
        preds, targets = model.predict(data_loader=test_loader)
        mean_squared_error = criterion(preds, targets).item()  # / torch.numel(preds)
        extrapolation_metrics[f"cutoff {cutoff}"] = {"MSE": mean_squared_error}

        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        mean_absolute_errors = np.mean(np.abs(preds - targets), axis=(0, 2))
        errors[cutoffs == cutoff] = mean_absolute_errors
        mean_absolute_error = np.mean(mean_absolute_errors)
        extrapolation_metrics[f"cutoff {cutoff}"]["MAE"] = mean_absolute_error

    # Extract metrics and errors for plotting
    model_errors = np.array(
        # [metric["MSE"] for metric in extrapolation_metrics.values()]
        [metric["MAE"] for metric in extrapolation_metrics.values()]
    )
    extrapolation_metrics["model_errors"] = model_errors
    extrapolation_metrics["cutoffs"] = cutoffs

    # Plot extrapolation errors
    plot_generalization_errors(
        surr_name, conf, cutoffs, model_errors, mode="extrapolation", save=True
    )
    plot_average_errors_over_time(
        surr_name,
        conf,
        errors,
        cutoffs,
        timesteps,
        mode="extrapolation",
        save=True,
    )

    return extrapolation_metrics


def evaluate_sparse(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    n_train_samples: int,
    conf: dict,
) -> dict[str, Any]:
    """
    Evaluate the performance of the surrogate model with sparse training data.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        n_train_samples (int): The number of training samples in the full dataset.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing sparse training metrics.
    """
    training_id = conf["training_id"]
    factors = conf["sparse"]["factors"]
    factors = np.sort(np.array(factors, dtype=int))
    factors = factors[factors > 1]
    factors = np.insert(factors, 0, 1)
    sparse_metrics = {}
    errors = np.zeros((len(factors), len(timesteps)))

    # Criterion for prediction loss
    criterion = torch.nn.MSELoss()

    # Evaluate models for each factor
    for factor in factors:
        # Ensure that the main model is loaded for factor 1
        model_id = (
            f"{surr_name.lower()}_main"
            if factor == 1
            else f"{surr_name.lower()}_sparse_{factor}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)
        preds, targets = model.predict(data_loader=test_loader)
        mean_squared_error = criterion(preds, targets).item()  # / torch.numel(preds)
        train_samples = n_train_samples // factor
        sparse_metrics[f"factor {factor}"] = {
            "MSE": mean_squared_error,
            "n_train_samples": train_samples,
        }

        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        mean_absolute_errors = np.mean(np.abs(preds - targets), axis=(0, 2))
        errors[factors == factor] = mean_absolute_errors
        mean_absolute_error = np.mean(mean_absolute_errors)
        sparse_metrics[f"factor {factor}"]["MAE"] = mean_absolute_error

    # Extract metrics and errors for plotting
    # model_errors = np.array([metric["MSE"] for metric in sparse_metrics.values()])
    model_errors = np.array([metric["MAE"] for metric in sparse_metrics.values()])
    n_train_samples_array = np.array(
        [metric["n_train_samples"] for metric in sparse_metrics.values()]
    )
    sparse_metrics["model_errors"] = model_errors
    sparse_metrics["n_train_samples"] = n_train_samples_array

    # Plot sparse training errors
    plot_generalization_errors(
        surr_name, conf, n_train_samples_array, model_errors, mode="sparse", save=True
    )
    plot_average_errors_over_time(
        surr_name,
        conf,
        errors,
        n_train_samples_array,
        timesteps,
        mode="sparse",
        save=True,
    )

    return sparse_metrics


def evaluate_batchsize(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    conf: dict,
) -> dict[str, Any]:
    """
    Evaluate the performance of the surrogate model with different batch sizes.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        timesteps (np.ndarray): The timesteps array.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing batch size training metrics.
    """
    training_id = conf["training_id"]
    batch_sizes = conf["batch_scaling"]["sizes"]
    batch_metrics = {}
    errors = np.zeros((len(batch_sizes), len(timesteps)))

    # Criterion for prediction loss
    criterion = torch.nn.MSELoss()

    # Evaluate models for each batch size
    for i, batch_size in enumerate(batch_sizes):
        model_id = f"{surr_name.lower()}_batchsize_{batch_size}"
        model.load(training_id, surr_name, model_identifier=model_id)
        preds, targets = model.predict(data_loader=test_loader)
        mean_squared_error = criterion(preds, targets).item()  # / torch.numel(preds)
        batch_metrics[f"batch_size {batch_size}"] = {"MSE": mean_squared_error}

        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        mean_relative_errors = np.mean(np.abs((preds - targets) / targets), axis=(0, 2))
        errors[i] = mean_relative_errors

    # Extract metrics and errors for plotting
    model_errors = np.array([metric["MSE"] for metric in batch_metrics.values()])
    batch_sizes_array = np.array(batch_sizes)
    batch_metrics["model_errors"] = model_errors
    batch_metrics["batch_sizes"] = batch_sizes_array

    # Plot batch size training errors
    plot_generalization_errors(
        surr_name, conf, batch_sizes_array, model_errors, mode="batchsize", save=True
    )
    plot_average_errors_over_time(
        surr_name,
        conf,
        errors,
        batch_sizes_array,
        timesteps,
        mode="batchsize",
        save=True,
    )

    return batch_metrics


def evaluate_UQ(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    conf: dict,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate the uncertainty quantification (UQ) performance of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        timesteps (np.ndarray): The timesteps array.
        conf (dict): The configuration dictionary.
        labels (list, optional): The labels for the chemical species.

    Returns:
        dict: A dictionary containing UQ metrics.
    """
    training_id = conf["training_id"]
    n_models = conf["uncertainty"]["ensemble_size"]
    UQ_metrics = []

    # Obtain predictions for each model
    all_predictions = []
    for i in range(n_models):
        model_id = (
            f"{surr_name.lower()}_main" if i == 0 else f"{surr_name.lower()}_UQ_{i}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)
        preds, targets = model.predict(data_loader=test_loader)
        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        all_predictions.append(preds)

    all_predictions = np.array(all_predictions)

    # Calculate average uncertainty
    preds_mean = np.mean(all_predictions, axis=0)
    preds_std = np.std(all_predictions, axis=0, ddof=1)
    average_uncertainty = np.mean(preds_std)

    # Correlate uncertainty with errors
    errors = np.abs(preds_mean - targets)
    errors_time = np.mean(errors, axis=(0, 2))
    avg_correlation, _ = pearsonr(errors.flatten(), preds_std.flatten())
    preds_std_time = np.mean(preds_std, axis=(0, 2))

    # Plots
    plot_example_predictions_with_uncertainty(
        surr_name,
        conf,
        preds_mean,
        preds_std,
        targets,
        timesteps,
        save=True,
        labels=labels,
    )
    plot_average_uncertainty_over_time(
        surr_name, conf, errors_time, preds_std_time, timesteps, save=True
    )
    # plot_uncertainty_vs_errors(surr_name, conf, preds_std, errors, save=True)
    max_counts, axis_max = plot_error_correlation_heatmap(
        surr_name, conf, preds_std, errors, avg_correlation, save=True
    )

    # Store metrics
    UQ_metrics = {
        "average_uncertainty": average_uncertainty,
        "correlation_metrics": avg_correlation,
        "pred_uncertainty": preds_std,
        "max_counts": max_counts,
        "axis_max": axis_max,
    }

    return UQ_metrics


def compare_models(metrics: dict, config: dict):

    print("Making comparative plots... \n")

    # Compare relative errors
    compare_relative_errors(metrics, config)
    compare_MAE(metrics, config)
    if config["losses"]:
        compare_main_losses(metrics, config)

    if config["gradients"]:
        compare_dynamic_accuracy(metrics, config)

    # Compare inference time
    if config["timing"]:
        compare_inference_time(metrics, config)

    # Compare interpolation errors
    if config["interpolation"]["enabled"]:
        compare_interpolation(metrics, config)

    # Compare extrapolation errors
    if config["extrapolation"]["enabled"]:
        compare_extrapolation(metrics, config)

    # Compare sparse training errors
    if config["sparse"]["enabled"]:
        compare_sparse(metrics, config)

    if (
        config["interpolation"]["enabled"]
        and config["extrapolation"]["enabled"]
        and config["sparse"]["enabled"]
    ):
        int_ext_sparse(metrics, config)

    # Compare batch size training errors
    if config["batch_scaling"]["enabled"]:
        compare_batchsize(metrics, config)

    # Compare UQ metrics
    if config["uncertainty"]["enabled"]:
        compare_UQ(metrics, config)
        rel_errors_and_uq(metrics, config)

    tabular_comparison(metrics, config)


def compare_main_losses(metrics: dict, config: dict) -> None:
    """
    Compare the training and test losses of the main models for different surrogate models.

    Args:
        metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    train_losses = []
    test_losses = []
    labels = []
    device = config["devices"]
    device = device[0] if isinstance(device, list) else device

    for surr_name, _ in metrics.items():
        training_id = config["training_id"]
        surrogate_class = get_surrogate(surr_name)
        n_timesteps = metrics[surr_name]["timesteps"].shape[0]
        n_chemicals = metrics[surr_name]["accuracy"]["absolute_errors"].shape[2]
        model_config = get_model_config(surr_name, config)
        model = surrogate_class(device, n_chemicals, n_timesteps, model_config)

        def load_losses(model_identifier: str):
            model.load(training_id, surr_name, model_identifier=model_identifier)
            return model.train_loss, model.test_loss

        # Load main model losses
        main_train_loss, main_test_loss = load_losses(f"{surr_name.lower()}_main")
        train_losses.append(main_train_loss)
        test_losses.append(main_test_loss)
        labels.append(surr_name)

    # Plot the comparison of main model losses
    plot_loss_comparison(tuple(train_losses), tuple(test_losses), tuple(labels), config)


def compare_MAE(metrics: dict, config: dict) -> None:
    """
    Compare the MAE of different surrogate models over the course of training.

    Args:
        metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    MAE = []
    labels = []
    train_durations = []
    device = config["devices"]
    device = device[0] if isinstance(device, list) else device

    for surr_name, _ in metrics.items():
        training_id = config["training_id"]
        surrogate_class = get_surrogate(surr_name)
        n_timesteps = metrics[surr_name]["timesteps"].shape[0]
        n_chemicals = metrics[surr_name]["accuracy"]["absolute_errors"].shape[2]
        model_config = get_model_config(surr_name, config)
        model = surrogate_class(device, n_chemicals, n_timesteps, model_config)
        model_identifier = f"{surr_name.lower()}_main"
        model.load(training_id, surr_name, model_identifier=model_identifier)
        MAE.append(model.MAE)
        labels.append(surr_name)
        train_durations.append(model.train_duration)

    # plot_MAE_comparison(MAE, labels, config)
    plot_MAE_comparison_train_duration(MAE, labels, train_durations, config)


def compare_relative_errors(metrics: dict[str, dict], config: dict) -> None:
    """
    Compare the relative errors over time for different surrogate models.

    Args:
        metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    errors = {}
    mean_errors = {}
    median_errors = {}

    for surrogate, surrogate_metrics in metrics.items():
        relative_error_model = surrogate_metrics["accuracy"].get("relative_errors")
        if relative_error_model is not None:
            errors[surrogate] = relative_error_model
        mean_error_model = np.mean(relative_error_model, axis=(0, 2))
        median_error_model = np.median(relative_error_model, axis=(0, 2))
        if mean_error_model is not None and median_error_model is not None:
            mean_errors[surrogate] = mean_error_model
            median_errors[surrogate] = median_error_model
            timesteps = surrogate_metrics["timesteps"]

    plot_relative_errors(mean_errors, median_errors, timesteps, config)

    plot_error_distribution_comparative(errors, config)


def compare_inference_time(
    metrics: dict[str, dict], config: dict, save: bool = True
) -> None:
    """
    Compare the mean inference time of different surrogate models.

    Args:
        metrics (dict[str, dict]): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.
        save (bool, optional): Whether to save the plot. Defaults to True.

    Returns:
        None
    """
    mean_inference_times = {}
    std_inference_times = {}

    for surrogate, surrogate_metrics in metrics.items():
        if "timing" in surrogate_metrics:
            mean_time = surrogate_metrics["timing"].get("mean_inference_time_per_run")
            std_time = surrogate_metrics["timing"].get("std_inference_time_per_run")
            if mean_time is not None and std_time is not None:
                mean_inference_times[surrogate] = mean_time
                std_inference_times[surrogate] = std_time

    surrogates = list(mean_inference_times.keys())
    means = list(mean_inference_times.values())
    stds = list(std_inference_times.values())

    inference_time_bar_plot(surrogates, means, stds, config, save)


def compare_dynamic_accuracy(metrics: dict, config: dict) -> None:
    """
    Compare the gradients of different surrogate models.

    Args:
        metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    gradients = {}
    abs_errors = {}
    max_grads = {}
    max_errors = {}
    max_counts = {}
    corrs = {}

    for surrogate, surrogate_metrics in metrics.items():
        gradients[surrogate] = surrogate_metrics["gradients"]["gradients"]
        abs_errors[surrogate] = surrogate_metrics["accuracy"]["absolute_errors"]
        max_grads[surrogate] = surrogate_metrics["gradients"]["max_gradient"]
        max_errors[surrogate] = surrogate_metrics["gradients"]["max_error"]
        max_counts[surrogate] = surrogate_metrics["gradients"]["max_counts"]
        corrs[surrogate] = surrogate_metrics["gradients"]["avg_correlation"]

    plot_comparative_dynamic_correlation_heatmaps(
        gradients, abs_errors, corrs, max_grads, max_errors, max_counts, config
    )


def compare_interpolation(all_metrics: dict, config: dict) -> None:
    """
    Compare the interpolation errors of different surrogate models.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    surrogates = list(all_metrics.keys())
    intervals = []
    model_errors = []

    for surrogate in surrogates:
        if "interpolation" in all_metrics[surrogate]:
            intervals.append(all_metrics[surrogate]["interpolation"]["intervals"])
            model_errors.append(all_metrics[surrogate]["interpolation"]["model_errors"])

    plot_generalization_error_comparison(
        surrogates,
        intervals,
        model_errors,
        "Interpolation Interval",
        "interpolation_errors.png",
        config,
    )


def compare_extrapolation(all_metrics: dict, config: dict) -> None:
    """
    Compare the extrapolation errors of different surrogate models.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    surrogates = list(all_metrics.keys())
    cutoffs = []
    model_errors = []

    for surrogate in surrogates:
        if "extrapolation" in all_metrics[surrogate]:
            cutoffs.append(all_metrics[surrogate]["extrapolation"]["cutoffs"])
            model_errors.append(all_metrics[surrogate]["extrapolation"]["model_errors"])

    plot_generalization_error_comparison(
        surrogates,
        cutoffs,
        model_errors,
        "Extrapolation Cutoff",
        "extrapolation_errors.png",
        config,
    )


def compare_sparse(all_metrics: dict, config: dict) -> None:
    """
    Compare the sparse training errors of different surrogate models.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    surrogates = list(all_metrics.keys())
    n_train_samples = []
    model_errors = []

    for surrogate in surrogates:
        if "sparse" in all_metrics[surrogate]:
            n_train_samples.append(all_metrics[surrogate]["sparse"]["n_train_samples"])
            model_errors.append(all_metrics[surrogate]["sparse"]["model_errors"])

    plot_generalization_error_comparison(
        surrogates,
        n_train_samples,
        model_errors,
        "Number of Training Samples",
        "sparse_errors.png",
        config,
        xlog=True,
    )


def compare_batchsize(all_metrics: dict, config: dict) -> None:
    """
    Compare the batch size training errors of different surrogate models.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    surrogates = list(all_metrics.keys())
    batch_sizes = []
    model_errors = []

    for surrogate in surrogates:
        if "batch_size" in all_metrics[surrogate]:
            batch_sizes.append(all_metrics[surrogate]["batch_size"]["batch_sizes"])
            model_errors.append(all_metrics[surrogate]["batch_size"]["model_errors"])

    plot_generalization_error_comparison(
        surrogates,
        batch_sizes,
        model_errors,
        "Batch Size",
        "batch_size_errors.png",
        config,
        xlog=True,
    )


def compare_UQ(all_metrics: dict, config: dict) -> None:
    """
    Compare the uncertainty quantification (UQ) metrics of different surrogate models.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    pred_unc = {}
    pred_unc_time = {}
    abs_errors = {}
    axis_max = {}
    max_counts = {}
    corrs = {}
    for surrogate, surrogate_metrics in all_metrics.items():
        timesteps = surrogate_metrics["timesteps"]
        abs_errors[surrogate] = surrogate_metrics["accuracy"]["absolute_errors"]
        pred_unc[surrogate] = surrogate_metrics["UQ"]["pred_uncertainty"]
        pred_unc_time[surrogate] = np.mean(pred_unc[surrogate], axis=(0, 2))
        axis_max[surrogate] = surrogate_metrics["UQ"]["axis_max"]
        max_counts[surrogate] = surrogate_metrics["UQ"]["max_counts"]
        corrs[surrogate] = surrogate_metrics["UQ"]["correlation_metrics"]

    plot_uncertainty_over_time_comparison(pred_unc_time, abs_errors, timesteps, config)

    plot_comparative_error_correlation_heatmaps(
        pred_unc, abs_errors, corrs, axis_max, max_counts, config
    )


def tabular_comparison(all_metrics: dict, config: dict) -> None:
    """
    Compare the metrics of different surrogate models in a tabular format.

    Args:
        all_metrics (dict): dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    print("The results are in! Here is a summary of the benchmark metrics:\n")

    # Initialize the table headers and rows
    model_names = list(all_metrics.keys())
    headers = ["Metric"] + model_names
    rows = []

    # Accuracy metrics (always included)
    mse_values = [
        metrics["accuracy"]["mean_squared_error"] for metrics in all_metrics.values()
    ]
    mae_values = [
        metrics["accuracy"]["mean_absolute_error"] for metrics in all_metrics.values()
    ]
    mre_values = [
        metrics["accuracy"]["mean_relative_error"] for metrics in all_metrics.values()
    ]
    epochs = [
        metrics["accuracy"]["main_model_epochs"] for metrics in all_metrics.values()
    ]
    train_times = [
        int(metrics["accuracy"]["main_model_training_time"])
        for metrics in all_metrics.values()
    ]

    # Find the best (minimum) MSE and MRE values
    best_mse_index = np.argmin(mse_values)
    best_mae_index = np.argmin(mae_values)
    best_mre_index = np.argmin(mre_values)
    best_time_index = np.argmin(train_times)

    mse_row = ["MSE"] + [
        f"{value:.2e}" if i != best_mse_index else f"* {value:.2e} *"
        for i, value in enumerate(mse_values)
    ]
    mae_row = ["MAE"] + [
        f"{value:.2e}" if i != best_mae_index else f"* {value:.2e} *"
        for i, value in enumerate(mae_values)
    ]
    mre_row = ["MRE"] + [
        f"{value * 100:.2f} %" if i != best_mre_index else f"* {value * 100:.2f} % *"
        for i, value in enumerate(mre_values)
    ]
    epochs_row = ["Epochs"] + [value for i, value in enumerate(epochs)]
    train_strings = [f"{format_seconds(time)}" for time in train_times]
    tt_row = ["Train Time (hh:mm:ss)"] + [
        f"{time}" if i != best_time_index else f"* {time} *"
        for i, time in enumerate(train_strings)
    ]
    rows.extend([mse_row, mae_row, mre_row, epochs_row, tt_row])

    # Timing metrics (if enabled)
    if config.get("timing", False):
        mean_times = [
            metrics["timing"]["mean_inference_time_per_run"]
            for metrics in all_metrics.values()
        ]
        std_times = [
            metrics["timing"]["std_inference_time_per_run"]
            for metrics in all_metrics.values()
        ]

        # Find the best (minimum) inference time
        best_time_index = np.argmin(mean_times)

        timing_row = ["Inference Times"] + [
            (
                f"{format_time(mean, std)}"
                if i != best_time_index
                else f"* {format_time(mean, std)} *"
            )
            for i, (mean, std) in enumerate(zip(mean_times, std_times))
        ]
        rows.append(timing_row)

    # Gradients (if enabled)
    if config.get("gradients", False):
        avg_corr_values = [
            metrics["gradients"]["avg_correlation"] for metrics in all_metrics.values()
        ]
        avg_corr_row = ["Gradient Corr."] + [
            f"{value:.4f}" for value in avg_corr_values
        ]
        rows.append(avg_corr_row)

    # Compute metrics (if enabled)
    if config.get("compute", False):
        num_params_row = ["# Trainable Params"] + [
            f'{metrics["compute"]["num_trainable_parameters"]}'
            for metrics in all_metrics.values()
        ]
        # memory_row = ["Memory Footprint (MB)"] + [
        #     f'{(metrics["compute"]["memory_footprint"]["forward_memory_nograd"]/1e6):.2f}'
        #     for metrics in all_metrics.values()
        # ]
        rows.extend([num_params_row])

    # UQ metrics (if enabled)
    if config.get("uncertainty", False).get("enabled", False):
        avg_uncertainties = [
            metrics["UQ"]["average_uncertainty"] for metrics in all_metrics.values()
        ]
        uq_corr_values = [
            metrics["UQ"]["correlation_metrics"] for metrics in all_metrics.values()
        ]
        avg_unc_row = ["Avg. Uncertainty"] + [
            f"{value:.2e}" for value in avg_uncertainties
        ]
        uq_corr_row = ["UQ Corr."] + [f"{value:.4f}" for value in uq_corr_values]
        rows.extend([avg_unc_row, uq_corr_row])

    # Print the table using tabulate
    table = tabulate(rows, headers, tablefmt="simple_grid")
    print(table)
    print()

    # Save the table to a file if an output file path is provided
    txt_path = f"results/{config['training_id']}/metrics_table.txt"
    with open(txt_path, "w") as f:
        with redirect_stdout(f):
            print(table)

    # Make a csv file that contains all metrics for each model
    make_comparison_csv(metrics=all_metrics, config=config)
