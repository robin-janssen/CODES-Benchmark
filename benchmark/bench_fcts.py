import os
import torch
import numpy as np
from typing import Dict, Any
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from surrogates.DeepONet.dataloader import create_dataloader_chemicals

from .bench_plots import (
    plot_relative_errors_over_time,
    plot_dynamic_correlation,
    plot_generalization_errors,
    plot_sparse_errors,
)
from .bench_utils import load_model
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
    surr = surrogate_class()

    # Placeholder for metrics
    metrics = {}
    training_id = conf["training_ID"]
    base_dir = os.path.join("trained", surr_name, training_id)

    full_test_data, timesteps, N_train_samples = check_and_load_data(
        conf["dataset"], "test"
    )

    # Create dataloader for the test data
    batch_size = surr.config.batch_size
    test_loader = create_dataloader_chemicals(
        full_test_data, timesteps, batch_size=batch_size, shuffle=False
    )

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
        n_models = conf["UQ"]["n_models"]
        uq_metrics = []
        for i in range(n_models):
            statedict_path = os.path.join(base_dir, f"{surr_name.lower()}_UQ_{i}.pth")
            model = load_model(surr, statedict_path)
            uq_metrics.append(evaluate_UQ(model, conf))
        metrics["UQ"] = uq_metrics

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
    training_id = conf["training_ID"]

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
    training_id = conf["training_ID"]

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

    # Store metrics
    dynamic_accuracy_metrics = {
        "species_correlations": species_correlations,
        "avg_correlation": avg_correlation,
    }

    return dynamic_accuracy_metrics


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
    training_id = conf["training_ID"]
    intervals = conf["interpolation"]["intervals"]
    interpolation_metrics = []

    # Criterion for prediction loss
    criterion = torch.nn.MSELoss(reduction="sum")

    # Evaluate main model (interval 1)
    model = load_model(
        surr, training_id, surr_name, model_identifier=f"{surr_name.lower()}_main"
    )
    total_loss, _, _ = model.predict(
        test_loader,
        criterion,
        N_timesteps=len(timesteps),
    )
    interpolation_metrics.append({"interval": 1, "total_loss": total_loss})

    # Evaluate models for each interval
    for interval in intervals:
        model = load_model(
            surr,
            training_id,
            surr_name,
            model_identifier=f"{surr_name.lower()}_interpolation_{interval}",
        )
        total_loss, _, _ = model.predict(
            test_loader, criterion, N_timesteps=len(timesteps)
        )
        interpolation_metrics.append({"interval": interval, "total_loss": total_loss})

    # Extract metrics and errors for plotting
    metrics = np.array([metric["interval"] for metric in interpolation_metrics])
    model_errors = np.array([metric["total_loss"] for metric in interpolation_metrics])

    # Plot interpolation errors
    plot_generalization_errors(
        surr_name, conf, metrics, model_errors, interpolate=True, save=True
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
    training_id = conf["training_ID"]
    cutoffs = conf["extrapolation"]["cutoffs"]
    extrapolation_metrics = []

    # Criterion for prediction loss
    criterion = torch.nn.MSELoss(reduction="sum")

    # Evaluate main model (no cutoff)
    model = load_model(
        surr, training_id, surr_name, model_identifier=f"{surr_name.lower()}_main"
    )
    total_loss, _, _ = model.predict(test_loader, criterion, N_timesteps=len(timesteps))
    extrapolation_metrics.append({"cutoff": len(timesteps), "total_loss": total_loss})

    # Evaluate models for each cutoff
    for cutoff in cutoffs:
        model = load_model(
            surr,
            training_id,
            surr_name,
            model_identifier=f"{surr_name.lower()}_extrapolation_{cutoff}",
        )
        total_loss, _, _ = model.predict(
            test_loader, criterion, N_timesteps=len(timesteps)
        )
        extrapolation_metrics.append({"cutoff": cutoff, "total_loss": total_loss})

    # Extract metrics and errors for plotting
    metrics = np.array([metric["cutoff"] for metric in extrapolation_metrics])
    model_errors = np.array([metric["total_loss"] for metric in extrapolation_metrics])

    # Plot extrapolation errors
    plot_generalization_errors(
        surr_name, conf, metrics, model_errors, interpolate=False, save=True
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
    training_id = conf["training_ID"]
    factors = conf["sparse"]["factors"]
    sparse_metrics = []

    # Criterion for prediction loss
    criterion = torch.nn.MSELoss(reduction="sum")

    # Evaluate main model (factor 1)
    model = load_model(
        surr, training_id, surr_name, model_identifier=f"{surr_name.lower()}_main"
    )
    total_loss, _, _ = model.predict(
        test_loader,
        criterion,
        N_timesteps=len(test_loader.dataset),
    )
    sparse_metrics.append(
        {"factor": 1, "total_loss": total_loss, "n_train_samples": N_train_samples}
    )

    # Evaluate models for each factor
    for factor in factors:
        model = load_model(
            surr,
            training_id,
            surr_name,
            model_identifier=f"{surr_name.lower()}_sparse_{factor}",
        )
        total_loss, _, _ = model.predict(
            test_loader,
            criterion,
            N_timesteps=len(test_loader.dataset),
        )
        n_train_samples = N_train_samples // factor
        sparse_metrics.append(
            {
                "factor": factor,
                "total_loss": total_loss,
                "n_train_samples": n_train_samples,
            }
        )

    # Extract metrics and errors for plotting
    factors = np.array([metric["factor"] for metric in sparse_metrics])
    model_errors = np.array([metric["total_loss"] for metric in sparse_metrics])
    n_train_samples_array = np.array(
        [metric["n_train_samples"] for metric in sparse_metrics]
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


def evaluate_UQ(model, conf: Dict) -> Dict[str, Any]:
    """
    Evaluate the uncertainty quantification of the surrogate model.

    Args:
        model: The surrogate model to evaluate.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing UQ metrics.
    """
    # Placeholder for UQ evaluation logic
    pass


def compare_models(metrics):
    # Compare models
    pass
