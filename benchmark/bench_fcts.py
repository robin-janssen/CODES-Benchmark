import os
import torch
import numpy as np
from typing import Dict, Any
from torch.utils.data import DataLoader

from surrogates.DeepONet.dataloader import create_dataloader_chemicals

from .bench_plots import plot_relative_errors_over_time


def run_benchmark(surrogate_name: str, surrogate_class, conf: Dict) -> Dict[str, Any]:
    """
    Run benchmarks for a given surrogate model.

    Args:
        surrogate_name (str): The name of the surrogate model to benchmark.
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
    base_dir = os.path.join("trained", surrogate_name, training_id)

    # Load data
    data_path = conf["data_path"]
    full_test_data = np.load(os.path.join(data_path, "test_data.npy"))
    osu_timesteps = np.linspace(0, 99, 100)

    # Create dataloader for the test data
    batch_size = surr.config.batch_size
    dataloader_test = create_dataloader_chemicals(
        full_test_data, osu_timesteps, batch_size=batch_size, shuffle=False
    )

    # Accuracy benchmark
    if conf["accuracy"]:
        print("Running accuracy benchmark...")
        statedict_path = os.path.join(
            base_dir, f"accuracy_{surrogate_name.lower()}.pth"
        )
        model = load_model(surr, statedict_path)
        metrics["accuracy"] = evaluate_accuracy(model, dataloader_test, conf)

    # Dynamic accuracy benchmark
    if conf["dynamic_accuracy"]:
        print("Running dynamic accuracy benchmark...")
        statedict_path = os.path.join(
            base_dir, f"accuracy_{surrogate_name.lower()}.pth"
        )
        model = load_model(surr, statedict_path)
        metrics["dynamic_accuracy"] = evaluate_dynamic_accuracy(model, conf)

    # Interpolation benchmark
    if conf["interpolation"]["enabled"]:
        print("Running interpolation benchmark...")
        intervals = conf["interpolation"]["intervals"]
        interpolation_metrics = []
        for interval in intervals:
            statedict_path = os.path.join(
                base_dir, f"interpolation_{interval}_{surrogate_name.lower()}.pth"
            )
            model = load_model(surr, statedict_path)
            interpolation_metrics.append(evaluate_interpolation(model, conf, interval))
        metrics["interpolation"] = interpolation_metrics

    # Extrapolation benchmark
    if conf["extrapolation"]["enabled"]:
        print("Running extrapolation benchmark...")
        cutoffs = conf["extrapolation"]["cutoffs"]
        extrapolation_metrics = []
        for cutoff in cutoffs:
            statedict_path = os.path.join(
                base_dir, f"extrapolation_{cutoff}_{surrogate_name.lower()}.pth"
            )
            model = load_model(surr, statedict_path)
            extrapolation_metrics.append(evaluate_extrapolation(model, conf, cutoff))
        metrics["extrapolation"] = extrapolation_metrics

    # Sparse data benchmark
    if conf["sparse"]["enabled"]:
        print("Running sparse data benchmark...")
        factors = conf["sparse"]["factors"]
        sparse_metrics = []
        for factor in factors:
            statedict_path = os.path.join(
                base_dir, f"sparse_{factor}_{surrogate_name.lower()}.pth"
            )
            model = load_model(surr, statedict_path)
            sparse_metrics.append(evaluate_sparse(model, conf, factor))
        metrics["sparse"] = sparse_metrics

    # Uncertainty Quantification (UQ) benchmark
    if conf["UQ"]["enabled"]:
        print("Running UQ benchmark...")
        n_models = conf["UQ"]["n_models"]
        uq_metrics = []
        for i in range(n_models):
            statedict_path = os.path.join(
                base_dir, f"UQ_{i}_{surrogate_name.lower()}.pth"
            )
            model = load_model(surr, statedict_path)
            uq_metrics.append(evaluate_UQ(model, conf))
        metrics["UQ"] = uq_metrics

    return metrics


def load_model(model, statedict_path: str) -> torch.nn.Module:
    """
    Load a trained surrogate model.

    Args:
        statedict_path (str): The path to the model dictionary.
        model: An instance of the surrogate model class.

    Returns:
        The loaded surrogate model.
    """
    model.load_state_dict(statedict_path)
    model.eval()
    return model


def evaluate_accuracy(model, dataloader_test: DataLoader, conf: Dict) -> Dict[str, Any]:
    """
    Evaluate the accuracy of the surrogate model.

    Args:
        model: The surrogate model to evaluate.
        dataloader_test (DataLoader): The DataLoader object containing the test data.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing accuracy metrics.
    """
    # TODO: Test this function
    # Use the model's predict method
    criterion = torch.nn.MSELoss(reduction="sum")
    total_loss, preds_buffer, targets_buffer = model.predict(
        dataloader_test, criterion, N_timesteps=100, reshape=True, transpose=True
    )

    # Calculate relative errors
    relative_errors = np.abs((preds_buffer - targets_buffer) / targets_buffer)

    # Plot relative errors over time
    plot_relative_errors_over_time(
        relative_errors,
        title=f"Relative Errors over Time for {model.__class__.__name__}",
        save=True,
        conf=conf,
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


def evaluate_dynamic_accuracy(model, conf: Dict) -> Dict[str, Any]:
    """
    Evaluate the dynamic accuracy of the surrogate model.

    Args:
        model: The surrogate model to evaluate.
        conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing dynamic accuracy metrics.
    """
    # Placeholder for dynamic accuracy evaluation logic
    pass


def evaluate_interpolation(model, conf: Dict, interval: int) -> Dict[str, Any]:
    """
    Evaluate the interpolation capability of the surrogate model.

    Args:
        model: The surrogate model to evaluate.
        conf (dict): The configuration dictionary.
        interval (int): The interpolation interval.

    Returns:
        dict: A dictionary containing interpolation metrics.
    """
    # Placeholder for interpolation evaluation logic
    pass


def evaluate_extrapolation(model, conf: Dict, cutoff: int) -> Dict[str, Any]:
    """
    Evaluate the extrapolation capability of the surrogate model.

    Args:
        model: The surrogate model to evaluate.
        conf (dict): The configuration dictionary.
        cutoff (int): The extrapolation cutoff.

    Returns:
        dict: A dictionary containing extrapolation metrics.
    """
    # Placeholder for extrapolation evaluation logic
    pass


def evaluate_sparse(model, conf: Dict, factor: int) -> Dict[str, Any]:
    """
    Evaluate the performance of the surrogate model with sparse data.

    Args:
        model: The surrogate model to evaluate.
        conf (dict): The configuration dictionary.
        factor (int): The sparsity factor.

    Returns:
        dict: A dictionary containing sparse data metrics.
    """
    # Placeholder for sparse data evaluation logic
    pass


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
