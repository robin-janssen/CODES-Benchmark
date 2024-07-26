import torch
import numpy as np
import time
import os
from typing import Dict, Any
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from .bench_plots import (
    plot_relative_errors_over_time,
    plot_dynamic_correlation,
    plot_generalization_errors,
    plot_average_errors_over_time,
    plot_average_uncertainty_over_time,
    plot_example_predictions_with_uncertainty,
    plot_uncertainty_vs_errors,
    plot_surr_losses,
    plot_loss_comparison,
    # plot_accuracy_comparison,
    plot_accuracy_comparison_train_duration,
    plot_relative_errors,
    inference_time_bar_plot,
    plot_generalization_error_comparison,
    plot_error_correlation_heatmap,
    plot_dynamic_correlation_heatmap,
)
from .bench_utils import (
    count_trainable_parameters,
    measure_memory_footprint,
    write_metrics_to_yaml,
    get_surrogate,
)
from data import check_and_load_data


def run_benchmark(surr_name: str, surrogate_class, conf: Dict) -> Dict[str, Any]:
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
    model = surrogate_class(device=device)

    # Placeholder for metrics
    metrics = {}

    full_train_data, full_test_data, full_val_data, timesteps, N_train_samples, _ = (
        check_and_load_data(
            conf["dataset"]["name"],
            verbose=False,
            log=conf["dataset"]["log10_transform"],
            normalisation_mode=conf["dataset"]["normalise"],
        )
    )

    # Create dataloader for the test data
    _, _, val_loader = model.prepare_data(
        dataset_train=full_train_data,
        dataset_test=full_test_data,
        dataset_val=full_val_data,
        timesteps=timesteps,
        shuffle=False,
    )

    # Plot training losses
    if conf["losses"]:
        plot_surr_losses(model, surr_name, conf, timesteps)

    # Accuracy benchmark
    if conf["accuracy"]:
        print("Running accuracy benchmark...")
        metrics["accuracy"] = evaluate_accuracy(
            model, surr_name, val_loader, timesteps, conf
        )

    # Dynamic accuracy benchmark
    if conf["dynamic_accuracy"]:
        print("Running dynamic accuracy benchmark...")
        # For this benchmark, we can also use the main model
        metrics["dynamic_accuracy"] = evaluate_dynamic_accuracy(
            model, surr_name, val_loader, timesteps, conf
        )

    # Timing benchmark
    if conf["timing"]:
        print("Running timing benchmark...")
        metrics["timing"] = time_inference(
            model, surr_name, val_loader, timesteps, conf
        )

    # Compute (resources) benchmark
    if conf["compute"]:
        print("Running compute benchmark...")
        metrics["compute"] = evaluate_compute(
            model, surr_name, val_loader, timesteps, conf
        )

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
            model, surr_name, val_loader, timesteps, N_train_samples, conf
        )

    # Batch size benchmark
    if conf["batch_scaling"]["enabled"]:
        print("Running batch size benchmark...")
        metrics["batch_size"] = evaluate_batchsize(
            model, surr_name, val_loader, timesteps, conf
        )

    # Uncertainty Quantification (UQ) benchmark
    if conf["UQ"]["enabled"]:
        print("Running UQ benchmark...")
        metrics["UQ"] = evaluate_UQ(model, surr_name, val_loader, timesteps, conf)

    # Write metrics to yaml
    write_metrics_to_yaml(surr_name, conf, metrics)

    return metrics


def evaluate_accuracy(
    model, surr_name: str, test_loader: DataLoader, timesteps, conf: Dict
) -> Dict[str, Any]:
    """
    Evaluate the accuracy of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing accuracy metrics.
    """
    training_id = conf["training_id"]

    # Load the model
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")

    # Use the model's predict method
    criterion = torch.nn.MSELoss(reduction="sum")
    preds, targets = model.predict(data_loader=test_loader, timesteps=timesteps)
    mean_squared_error = criterion(preds, targets).item() / torch.numel(preds)
    preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()

    # preds = preds.transpose(0, 2, 1)
    # targets = targets.transpose(0, 2, 1)

    # Calculate relative errors
    relative_errors = np.abs((preds - targets) / targets)

    # Calculate mean and median relative errors over time
    mean_relative_error = np.mean(relative_errors, axis=(0, 2))
    median_relative_error = np.median(relative_errors, axis=(0, 2))

    # Plot relative errors over time
    plot_relative_errors_over_time(
        surr_name,
        conf,
        relative_errors,
        title=f"Relative Errors over Time for {surr_name}",
        save=True,
    )

    # Store metrics
    accuracy_metrics = {
        "mean_squared_error": mean_squared_error,
        "mean_relative_error": np.mean(relative_errors),
        "median_relative_error": np.median(relative_errors),
        "max_relative_error": np.max(relative_errors),
        "min_relative_error": np.min(relative_errors),
        "mean_relative_error_over_time": mean_relative_error,
        "median_relative_error_over_time": median_relative_error,
    }

    return accuracy_metrics


def evaluate_dynamic_accuracy(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    conf: dict,
    species_names: list = None,
) -> dict:
    """
    Evaluate the dynamic accuracy of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        timesteps (np.ndarray): The timesteps array.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing dynamic accuracy metrics.
    """
    training_id = conf["training_id"]

    # Load the model
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")

    # Obtain predictions and targets
    preds, targets = model.predict(data_loader=test_loader, timesteps=timesteps)
    preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()

    # Calculate gradients of the target data w.r.t time
    gradients = np.gradient(targets, axis=1)
    # Take absolute value and normalize gradients
    gradients = np.abs(gradients) / np.abs(gradients).max()

    # Calculate absolute prediction errors
    prediction_errors = np.abs(preds - targets)
    # Normalize prediction errors
    prediction_errors = prediction_errors / prediction_errors.max()

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
    plot_dynamic_correlation(surr_name, conf, avg_gradient, avg_error, save=True)
    plot_dynamic_correlation_heatmap(
        surr_name, conf, gradients, prediction_errors, save=True
    )

    # Ensure species names are provided
    species_names = (
        species_names
        if species_names is not None
        else [f"quantity_{i}" for i in range(targets.shape[2])]
    )
    species_correlations = dict(zip(species_names, species_correlations))

    # Store metrics
    dynamic_accuracy_metrics = {
        "species_correlations": species_correlations,
        "avg_correlation": avg_correlation,
    }

    return dynamic_accuracy_metrics


def time_inference(
    model,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    conf: dict,
    n_runs: int = 5,
) -> Dict[str, Any]:
    """
    Time the inference of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        timesteps (np.ndarray): The timesteps array.
        conf (dict): The configuration dictionary.
        n_runs (int, optional): Number of times to run the inference for timing.

    Returns:
        dict: A dictionary containing timing metrics.
    """
    training_id = conf["training_id"]
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")

    # Run inference multiple times and record the durations
    inference_times = []
    for _ in range(n_runs):
        start_time = time.time()
        _, _ = model.predict(data_loader=test_loader, timesteps=timesteps)
        end_time = time.time()
        inference_times.append(end_time - start_time)

    # Calculate metrics
    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    num_predictions = len(test_loader.dataset)

    # Store metrics
    timing_metrics = {
        "mean_inference_time_per_run": mean_inference_time,
        "std_inference_time_per_run": std_inference_time,
        "num_predictions": num_predictions,
        "mean_inference_time_per_prediction": mean_inference_time / num_predictions,
        "std_inference_time_per_prediction": std_inference_time / num_predictions,
    }

    return timing_metrics


def evaluate_compute(
    model, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: Dict
) -> Dict[str, Any]:
    """
    Evaluate the computational resource requirements of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        timesteps (np.ndarray): The timesteps array.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing model complexity metrics.
    """
    training_id = conf["training_id"]
    model.load(training_id, surr_name, model_identifier=f"{surr_name.lower()}_main")

    # Count the number of trainable parameters
    num_params = count_trainable_parameters(model)

    # Get a sample input tensor from the test_loader
    inputs = next(iter(test_loader))
    # Measure the memory footprint during forward and backward pass
    memory_footprint = measure_memory_footprint(model, inputs, timesteps)

    # Store complexity metrics
    complexity_metrics = {
        "num_trainable_parameters": num_params,
        "memory_footprint": memory_footprint,
    }

    return complexity_metrics


def evaluate_interpolation(
    model, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: Dict
) -> Dict[str, Any]:
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
    criterion = torch.nn.MSELoss(reduction="sum")

    # Evaluate models for each interval
    for interval in intervals:
        # Ensure that the main model is loaded for interval 1
        model_id = (
            f"{surr_name.lower()}_main"
            if interval == 1
            else f"{surr_name.lower()}_interpolation_{interval}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)
        preds, targets = model.predict(data_loader=test_loader, timesteps=timesteps)
        mean_squared_error = criterion(preds, targets).item() / torch.numel(preds)
        interpolation_metrics[f"interval {interval}"] = {"MSE": mean_squared_error}

        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        mean_absolute_errors = np.mean(np.abs(preds - targets), axis=(0, 2))
        errors[intervals == interval] = mean_absolute_errors

    # Extract metrics and errors for plotting
    model_errors = np.array(
        [metric["MSE"] for metric in interpolation_metrics.values()]
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
    model, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: Dict
) -> Dict[str, Any]:
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
    cutoffs = np.insert(cutoffs, -1, max_cut)
    extrapolation_metrics = {}
    errors = np.zeros((len(cutoffs), len(timesteps)))

    # Criterion for prediction loss
    criterion = torch.nn.MSELoss(reduction="sum")

    # Evaluate models for each cutoff
    for cutoff in cutoffs:
        # Ensure that the main model is loaded for the last cutoff
        model_id = (
            f"{surr_name.lower()}_main"
            if cutoff == max_cut
            else f"{surr_name.lower()}_extrapolation_{cutoff}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)
        preds, targets = model.predict(data_loader=test_loader, timesteps=timesteps)
        mean_squared_error = criterion(preds, targets).item() / torch.numel(preds)
        extrapolation_metrics[f"cutoff {cutoff}"] = {"MSE": mean_squared_error}

        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        mean_absolute_errors = np.mean(np.abs(preds - targets), axis=(0, 2))
        errors[cutoffs == cutoff] = mean_absolute_errors

    # Extract metrics and errors for plotting
    model_errors = np.array(
        [metric["MSE"] for metric in extrapolation_metrics.values()]
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
    N_train_samples: int,
    conf: Dict,
) -> Dict[str, Any]:
    """
    Evaluate the performance of the surrogate model with sparse training data.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        N_train_samples (int): The number of training samples in the full dataset.
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
    criterion = torch.nn.MSELoss(reduction="sum")

    # Evaluate models for each factor
    for factor in factors:
        # Ensure that the main model is loaded for factor 1
        model_id = (
            f"{surr_name.lower()}_main"
            if factor == 1
            else f"{surr_name.lower()}_sparse_{factor}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)
        preds, targets = model.predict(data_loader=test_loader, timesteps=timesteps)
        mean_squared_error = criterion(preds, targets).item() / torch.numel(preds)
        n_train_samples = N_train_samples // factor
        sparse_metrics[f"factor {factor}"] = {
            "MSE": mean_squared_error,
            "n_train_samples": n_train_samples,
        }

        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        mean_absolute_errors = np.mean(np.abs(preds - targets), axis=(0, 2))
        errors[factors == factor] = mean_absolute_errors

    # Extract metrics and errors for plotting
    model_errors = np.array([metric["MSE"] for metric in sparse_metrics.values()])
    n_train_samples_array = np.array(
        [metric["n_train_samples"] for metric in sparse_metrics.values()]
    )
    sparse_metrics["model_errors"] = model_errors
    sparse_metrics["N_train_samples"] = n_train_samples_array

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
    conf: Dict,
) -> Dict[str, Any]:
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
    criterion = torch.nn.MSELoss(reduction="sum")

    # Evaluate models for each batch size
    for i, batch_size in enumerate(batch_sizes):
        model_id = f"{surr_name.lower()}_batchsize_{batch_size}"
        model.load(training_id, surr_name, model_identifier=model_id)
        preds, targets = model.predict(data_loader=test_loader, timesteps=timesteps)
        mean_squared_error = criterion(preds, targets).item() / torch.numel(preds)
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
    model, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: Dict
) -> Dict[str, Any]:
    """
    Evaluate the uncertainty quantification (UQ) performance of the surrogate model.

    Args:
        model: Instance of the surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        timesteps (np.ndarray): The timesteps array.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing UQ metrics.
    """
    training_id = conf["training_id"]
    n_models = conf["UQ"]["n_models"]
    UQ_metrics = []

    # Obtain predictions for each model
    all_predictions = []
    for i in range(n_models):
        model_id = (
            f"{surr_name.lower()}_main" if i == 0 else f"{surr_name.lower()}_UQ_{i}"
        )
        model.load(training_id, surr_name, model_identifier=model_id)
        preds, targets = model.predict(data_loader=test_loader, timesteps=timesteps)
        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        all_predictions.append(preds)

    all_predictions = np.array(all_predictions)

    # Calculate average uncertainty
    preds_mean = np.mean(all_predictions, axis=0)
    preds_std = np.std(all_predictions, axis=0)
    average_uncertainty = np.mean(preds_std)

    # Correlate uncertainty with errors
    errors = np.abs(preds_mean - targets)
    correlation_metrics, _ = pearsonr(errors.flatten(), preds_std.flatten())

    # Plots
    plot_example_predictions_with_uncertainty(
        surr_name, conf, preds_mean, preds_std, targets, timesteps, save=True
    )
    plot_average_uncertainty_over_time(surr_name, conf, preds_std, timesteps, save=True)
    plot_uncertainty_vs_errors(surr_name, conf, preds_std, errors, save=True)
    plot_error_correlation_heatmap(surr_name, conf, preds_std, errors, save=True)

    # Store metrics
    UQ_metrics = {
        "average_uncertainty": average_uncertainty,
        "correlation_metrics": correlation_metrics,
    }

    return UQ_metrics


def compare_models(metrics: dict, config: dict):

    # Compare accuracies
    if config["accuracy"]:
        compare_relative_errors(metrics, config)
        compare_accuracies(metrics, config)
        if config["losses"]:
            compare_main_losses(metrics, config)

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

    # Compare batch size training errors
    if config["batch_scaling"]["enabled"]:
        compare_batchsize(metrics, config)

    # Compare UQ metrics
    if config["UQ"]["enabled"]:
        compare_UQ(metrics, config)


def compare_main_losses(metrics: dict, config: dict) -> None:
    """
    Compare the training and test losses of the main models for different surrogate models.

    Args:
        metrics (dict): Dictionary containing the benchmark metrics for each surrogate model.
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
        model = surrogate_class(device=device)

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


def compare_accuracies(metrics: dict, config: dict) -> None:
    """
    Compare the accuracies of different surrogate models over the course of training.

    Args:
        metrics (dict): Dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    accuracies = []
    labels = []
    train_durations = []
    device = config["devices"]
    device = device[0] if isinstance(device, list) else device

    for surr_name, _ in metrics.items():
        training_id = config["training_id"]
        surrogate_class = get_surrogate(surr_name)
        model = surrogate_class(device=device)
        model_identifier = f"{surr_name.lower()}_main"
        model.load(training_id, surr_name, model_identifier=model_identifier)
        accuracies.append(model.accuracy)
        labels.append(surr_name)
        train_durations.append(model.train_duration)

    # plot_accuracy_comparison(accuracies, labels, config)
    plot_accuracy_comparison_train_duration(accuracies, labels, train_durations, config)


def compare_relative_errors(metrics: Dict[str, dict], config: dict) -> None:
    """
    Compare the relative errors over time for different surrogate models.

    Args:
        metrics (dict): Dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    mean_errors = {}
    median_errors = {}
    timesteps = None

    for surrogate, surrogate_metrics in metrics.items():
        if "accuracy" in surrogate_metrics:
            mean_error = surrogate_metrics["accuracy"].get(
                "mean_relative_error_over_time"
            )
            median_error = surrogate_metrics["accuracy"].get(
                "median_relative_error_over_time"
            )
            if mean_error is not None and median_error is not None:
                mean_errors[surrogate] = mean_error
                median_errors[surrogate] = median_error
                if timesteps is None:
                    timesteps = np.arange(len(mean_error))

    plot_relative_errors(mean_errors, median_errors, timesteps, config)


def compare_inference_time(
    metrics: Dict[str, Dict], config: Dict, save: bool = True
) -> None:
    """
    Compare the mean inference time of different surrogate models.

    Args:
        metrics (Dict[str, Dict]): Dictionary containing the benchmark metrics for each surrogate model.
        config (Dict): Configuration dictionary.
        save (bool, optional): Whether to save the plot. Defaults to True.

    Returns:
        None
    """
    mean_inference_times = {}
    std_inference_times = {}

    for surrogate, surrogate_metrics in metrics.items():
        if "timing" in surrogate_metrics:
            mean_time = surrogate_metrics["timing"].get(
                "mean_inference_time_per_prediction"
            )
            std_time = surrogate_metrics["timing"].get(
                "std_inference_time_per_prediction"
            )
            if mean_time is not None and std_time is not None:
                mean_inference_times[surrogate] = mean_time
                std_inference_times[surrogate] = std_time

    surrogates = list(mean_inference_times.keys())
    means = list(mean_inference_times.values())
    stds = list(std_inference_times.values())

    inference_time_bar_plot(surrogates, means, stds, config, save)


def compare_interpolation(all_metrics: dict, config: dict) -> None:
    """
    Compare the interpolation errors of different surrogate models.

    Args:
        all_metrics (dict): Dictionary containing the benchmark metrics for each surrogate model.
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
        all_metrics (dict): Dictionary containing the benchmark metrics for each surrogate model.
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
        all_metrics (dict): Dictionary containing the benchmark metrics for each surrogate model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    surrogates = list(all_metrics.keys())
    n_train_samples = []
    model_errors = []

    for surrogate in surrogates:
        if "sparse" in all_metrics[surrogate]:
            n_train_samples.append(all_metrics[surrogate]["sparse"]["N_train_samples"])
            model_errors.append(all_metrics[surrogate]["sparse"]["model_errors"])

    plot_generalization_error_comparison(
        surrogates,
        n_train_samples,
        model_errors,
        "Number of Training Samples",
        "sparse_errors.png",
        config,
    )


def compare_batchsize(all_metrics: dict, config: dict) -> None:
    """
    Compare the batch size training errors of different surrogate models.

    Args:
        all_metrics (dict): Dictionary containing the benchmark metrics for each surrogate model.
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
    pass
