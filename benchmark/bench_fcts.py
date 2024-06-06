import torch
import numpy as np
import time
from typing import Dict, Any
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from data import create_dataloader_chemicals

from .bench_plots import (
    plot_relative_errors_over_time,
    plot_dynamic_correlation,
    plot_generalization_errors,
    plot_sparse_errors,
    plot_average_uncertainty_over_time,
    plot_example_predictions_with_uncertainty,
    plot_uncertainty_vs_errors,
    plot_surr_losses,
)
from .bench_utils import (
    load_model,
    count_trainable_parameters,
    measure_memory_footprint,
    write_metrics_to_yaml,
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
    surr = surrogate_class(device=device)

    # Placeholder for metrics
    metrics = {}

    full_test_data, timesteps, N_train_samples = check_and_load_data(
        conf["dataset"], "test"
    )

    # Create dataloader for the test data
    batch_size = surr.config.batch_size
    test_loader = create_dataloader_chemicals(
        full_test_data, timesteps, batch_size=batch_size, shuffle=False
    )

    # Plot training losses
    if conf["losses"]:
        plot_surr_losses(surr_name, conf, timesteps)

    # Accuracy benchmark
    if conf["accuracy"]:
        print("Running accuracy benchmark...")
        metrics["accuracy"] = evaluate_accuracy(
            surr, surr_name, test_loader, timesteps, conf
        )

    # Dynamic accuracy benchmark
    if conf["dynamic_accuracy"]:
        print("Running dynamic accuracy benchmark...")
        # For this benchmark, we can also use the main model
        metrics["dynamic_accuracy"] = evaluate_dynamic_accuracy(
            surr, surr_name, test_loader, timesteps, conf
        )

    # Timing benchmark
    if conf["timing"]:
        print("Running timing benchmark...")
        metrics["timing"] = time_inference(
            surr, surr_name, test_loader, timesteps, conf
        )

    # Compute (resources) benchmark
    if conf["compute"]:
        print("Running compute benchmark...")
        metrics["compute"] = evaluate_compute(surr, surr_name, test_loader, conf)

    # Interpolation benchmark
    if conf["interpolation"]["enabled"]:
        print("Running interpolation benchmark...")
        metrics["interpolation"] = evaluate_interpolation(
            surr, surr_name, test_loader, timesteps, conf
        )

    # Extrapolation benchmark
    if conf["extrapolation"]["enabled"]:
        print("Running extrapolation benchmark...")
        metrics["extrapolation"] = evaluate_extrapolation(
            surr, surr_name, test_loader, timesteps, conf
        )

    # Sparse data benchmark
    if conf["sparse"]["enabled"]:
        print("Running sparse benchmark...")
        metrics["sparse"] = evaluate_sparse(
            surr, surr_name, test_loader, N_train_samples, conf
        )

    # Uncertainty Quantification (UQ) benchmark
    if conf["UQ"]["enabled"]:
        print("Running UQ benchmark...")
        metrics["UQ"] = evaluate_UQ(surr, surr_name, test_loader, timesteps, conf)

    # Write metrics to yaml
    write_metrics_to_yaml(surr_name, conf, metrics)

    return metrics


def evaluate_accuracy(
    surr, surr_name: str, test_loader: DataLoader, timesteps, conf: Dict
) -> Dict[str, Any]:
    """
    Evaluate the accuracy of the surrogate model.

    Args:
        surr: The surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing accuracy metrics.
    """
    training_id = conf["training_id"]

    # Load the model
    model = load_model(
        surr, training_id, surr_name, model_identifier=f"{surr_name.lower()}_main"
    )

    # Use the model's predict method
    criterion = torch.nn.MSELoss(reduction="sum")
    total_loss, preds_buffer, targets_buffer = model.predict(
        test_loader,
        criterion,
        N_timesteps=len(timesteps),
    )

    preds_buffer = preds_buffer.transpose(0, 2, 1)
    targets_buffer = targets_buffer.transpose(0, 2, 1)

    # Calculate relative errors
    relative_errors = np.abs((preds_buffer - targets_buffer) / targets_buffer)

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
        "total_loss": total_loss,
        "mean_relative_error": np.mean(relative_errors),
        "median_relative_error": np.median(relative_errors),
        "max_relative_error": np.max(relative_errors),
        "min_relative_error": np.min(relative_errors),
    }

    return accuracy_metrics


def evaluate_dynamic_accuracy(
    surr,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    conf: dict,
    species_names: list = None,
) -> dict:
    """
    Evaluate the dynamic accuracy of the surrogate model.

    Args:
        surr: The surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        timesteps (np.ndarray): The timesteps array.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing dynamic accuracy metrics.
    """
    training_id = conf["training_id"]

    # Load the model
    model = load_model(
        surr, training_id, surr_name, model_identifier=f"{surr_name.lower()}_main"
    )

    # Criterion for prediction loss
    criterion = torch.nn.MSELoss(reduction="sum")

    # Obtain predictions and targets
    _, preds_buffer, targets_buffer = model.predict(
        test_loader,
        criterion,
        N_timesteps=len(timesteps),
    )

    # Calculate gradients of the target data w.r.t time
    gradients = np.gradient(targets_buffer, axis=1)
    # Take absolute value and normalize gradients
    gradients = np.abs(gradients) / np.abs(gradients).max()

    # Calculate absolute prediction errors
    prediction_errors = np.abs(preds_buffer - targets_buffer)
    # Normalize prediction errors
    prediction_errors = prediction_errors / prediction_errors.max()

    # Calculate correlations
    species_correlations = []
    for i in range(targets_buffer.shape[2]):
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

    # Ensure species names are provided
    species_names = (
        species_names
        if species_names is not None
        else [f"quantity_{i}" for i in range(targets_buffer.shape[2])]
    )
    species_correlations = dict(zip(species_names, species_correlations))

    # Store metrics
    dynamic_accuracy_metrics = {
        "species_correlations": species_correlations,
        "avg_correlation": avg_correlation,
    }

    return dynamic_accuracy_metrics


def time_inference(
    surr,
    surr_name: str,
    test_loader: DataLoader,
    timesteps: np.ndarray,
    conf: dict,
    n_runs: int = 5,
) -> Dict[str, Any]:
    """
    Time the inference of the surrogate model.

    Args:
        surr: The surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        timesteps (np.ndarray): The timesteps array.
        conf (dict): The configuration dictionary.
        n_runs (int, optional): Number of times to run the inference for timing.

    Returns:
        dict: A dictionary containing timing metrics.
    """
    training_id = conf["training_id"]
    model_identifier = f"{surr_name.lower()}_main"
    model = load_model(surr, training_id, surr_name, model_identifier=model_identifier)

    criterion = torch.nn.MSELoss(reduction="sum")

    # Run inference multiple times and record the durations
    inference_times = []
    for _ in range(n_runs):
        start_time = time.time()
        _, _, _ = model.predict(test_loader, criterion, N_timesteps=len(timesteps))
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
    surr, surr_name: str, test_loader: DataLoader, conf: Dict
) -> Dict[str, Any]:
    """
    Evaluate the computational resource requirements of the surrogate model.

    Args:
        surr: The surrogate model class.
        surr_name (str): The name of the surrogate model.
        test_loader (DataLoader): The DataLoader object containing the test data.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing model complexity metrics.
    """
    training_id = conf["training_id"]
    model_identifier = f"{surr_name.lower()}_main"
    model = load_model(surr, training_id, surr_name, model_identifier=model_identifier)

    # Count the number of trainable parameters
    num_params = count_trainable_parameters(model)

    # Get a sample input tensor from the test_loader
    sample_ICs, sample_times, _ = next(iter(test_loader))
    # Measure the memory footprint during forward and backward pass
    memory_footprint = measure_memory_footprint(model, sample_ICs, sample_times)

    # Store complexity metrics
    complexity_metrics = {
        "num_trainable_parameters": num_params,
        "memory_footprint": memory_footprint,
    }

    return complexity_metrics


def evaluate_interpolation(
    surr, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: Dict
) -> Dict[str, Any]:
    """
    Evaluate the interpolation performance of the surrogate model.

    Args:
        surr: The surrogate model class.
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
        model = load_model(surr, training_id, surr_name, model_id)
        total_loss, _, _ = model.predict(
            test_loader, criterion, N_timesteps=len(timesteps)
        )
        interpolation_metrics[f"interval {interval}"] = {"MSE": total_loss}

    # Extract metrics and errors for plotting
    model_errors = np.array(
        [metric["MSE"] for metric in interpolation_metrics.values()]
    )

    # Plot interpolation errors
    plot_generalization_errors(
        surr_name, conf, intervals, model_errors, interpolate=True, save=True
    )

    return interpolation_metrics


def evaluate_extrapolation(
    surr, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: Dict
) -> Dict[str, Any]:
    """
    Evaluate the extrapolation performance of the surrogate model.

    Args:
        surr: The surrogate model class.
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
        model = load_model(surr, training_id, surr_name, model_id)
        total_loss, _, _ = model.predict(
            test_loader, criterion, N_timesteps=len(timesteps)
        )
        extrapolation_metrics[f"cutoff {cutoff}"] = {"MSE": total_loss}

    # Extract metrics and errors for plotting
    model_errors = np.array(
        [metric["MSE"] for metric in extrapolation_metrics.values()]
    )

    # Plot extrapolation errors
    plot_generalization_errors(
        surr_name, conf, cutoffs, model_errors, interpolate=False, save=True
    )

    return extrapolation_metrics


def evaluate_sparse(
    surr, surr_name: str, test_loader: DataLoader, N_train_samples: int, conf: Dict
) -> Dict[str, Any]:
    """
    Evaluate the performance of the surrogate model with sparse training data.

    Args:
        surr: The surrogate model class.
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
        model = load_model(surr, training_id, surr_name, model_id)
        total_loss, _, _ = model.predict(
            test_loader,
            criterion,
            N_timesteps=len(test_loader.dataset),
        )
        n_train_samples = N_train_samples // factor
        sparse_metrics[f"factor {factor}"] = {
            "MSE": total_loss,
            "n_train_samples": n_train_samples,
        }

    # Extract metrics and errors for plotting
    model_errors = np.array([metric["MSE"] for metric in sparse_metrics.values()])
    n_train_samples_array = np.array(
        [metric["n_train_samples"] for metric in sparse_metrics.values()]
    )

    # Plot sparse training errors
    plot_sparse_errors(
        surr_name,
        conf,
        n_train_samples_array,
        model_errors,
        title="Sparse Training Errors",
        save=True,
    )

    return sparse_metrics


def evaluate_UQ(
    surr, surr_name: str, test_loader: DataLoader, timesteps: np.ndarray, conf: Dict
) -> Dict[str, Any]:
    """
    Evaluate the uncertainty quantification (UQ) performance of the surrogate model.

    Args:
        surr: The surrogate model class.
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

    # Criterion for prediction loss
    criterion = torch.nn.MSELoss(reduction="sum")

    # Obtain predictions for each model
    all_predictions = []
    for i in range(n_models):
        model_identifier = (
            f"{surr_name.lower()}_main" if i == 0 else f"{surr_name.lower()}_UQ_{i}"
        )
        model = load_model(
            surr, training_id, surr_name, model_identifier=model_identifier
        )
        _, preds_buffer, targets_buffer = model.predict(
            test_loader, criterion, N_timesteps=len(timesteps)
        )
        all_predictions.append(preds_buffer)

    all_predictions = np.array(all_predictions)

    # Calculate average uncertainty
    preds_mean = np.mean(all_predictions, axis=0)
    preds_std = np.std(all_predictions, axis=0)
    average_uncertainty = np.mean(preds_std)

    # Correlate uncertainty with errors
    errors = np.abs(preds_mean - targets_buffer)
    correlation_metrics, _ = pearsonr(errors.flatten(), preds_std.flatten())

    # Plots
    plot_example_predictions_with_uncertainty(
        surr_name, conf, preds_mean, preds_std, targets_buffer, timesteps, save=True
    )
    plot_average_uncertainty_over_time(surr_name, conf, preds_std, timesteps, save=True)
    plot_uncertainty_vs_errors(surr_name, conf, preds_std, errors, save=True)

    # Store metrics
    UQ_metrics = {
        "average_uncertainty": average_uncertainty,
        "correlation_metrics": correlation_metrics,
    }

    return UQ_metrics


def compare_models(metrics):
    # Compare models
    pass
