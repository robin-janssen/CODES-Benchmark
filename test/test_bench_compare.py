import os

import numpy as np
import pytest

import codes.benchmark.bench_fcts as bf


# Helpers to build dummy metrics
def make_metrics_for_main(surr_names, n_timesteps=4, n_quantities=3):
    # for compare_main_losses
    metrics = {}
    for i, name in enumerate(surr_names):
        metrics[name] = {
            "timesteps": np.zeros(n_timesteps),
            "accuracy": {
                "absolute_errors": np.zeros((1, n_timesteps, n_quantities)),
                "absolute_errors_log": np.zeros((1, n_timesteps, n_quantities)),
            },
            "n_params": 123 + i,
        }
    return metrics


def make_metrics_for_relative(surr_names, timesteps):
    # build metrics with a relative_errors array
    metrics = {}
    for name in surr_names:
        rel = np.arange(len(timesteps) * 1.0 * 1.0).reshape(1, len(timesteps), 1)
        metrics[name] = {
            "accuracy": {"relative_errors": rel},
            "timesteps": np.array(timesteps),
        }
    return metrics


def make_metrics_for_timing(surr_names):
    metrics = {}
    for name in surr_names:
        metrics[name] = {
            "timing": {
                "mean_inference_time_per_run": 1.23,
                "std_inference_time_per_run": 0.45,
            }
        }
    return metrics


def make_metrics_for_dynamic(surr_names, n_timesteps=4, n_quantities=2):
    metrics = {}
    for name in surr_names:
        metrics[name] = {
            "accuracy": {
                "absolute_errors": np.zeros((1, n_timesteps, n_quantities)),
                "absolute_errors_log": np.zeros((1, n_timesteps, n_quantities)),
            },
            "gradients": {
                "gradients": np.ones((1, n_timesteps, n_quantities)),
                "avg_correlation": 0.5,
                "max_gradient": 0.6,
                "max_error": 0.7,
                "max_counts": 8,
            },
        }
    return metrics


def make_metrics_for_generalization(surr_names):
    base = {}
    for name in surr_names:
        base[name] = {
            "interpolation": {"intervals": [1, 2], "model_errors": [0.1, 0.2]},
            "extrapolation": {"cutoffs": [1, 3], "model_errors": [0.3, 0.4]},
            "sparse": {"n_train_samples": [5, 10], "model_errors": [0.5, 0.6]},
            "batch_size": {"batch_sizes": [16, 32], "model_errors": [0.7, 0.8]},
            "UQ": {
                "pred_uncertainty": np.ones((1, 2, 1)) * 0.9,
                "absolute_errors": np.zeros((1, 2, 1)),
                "relative_errors": np.zeros((1, 2, 1)),
                "axis_max": 2,
                "max_counts": 3,
                "correlation_metrics": 0.2,
                "weighted_diff": np.zeros((1, 2, 1)),
            },
        }
    return base


def make_metrics_for_interpolation(surr_names):
    base = {}
    for name in surr_names:
        base[name] = {
            "interpolation": {
                "intervals": np.array([1, 2, 4]),
                "model_errors": np.array([0.1, 0.2, 0.25]),
            }
        }
    return base


def make_metrics_for_extrapolation(surr_names, timesteps_len=5):
    base = {}
    for name in surr_names:
        base[name] = {
            "extrapolation": {
                "cutoffs": np.array([2, timesteps_len]),
                "model_errors": np.array([0.3, 0.22]),
            }
        }
    return base


def make_metrics_for_sparse(surr_names):
    base = {}
    for name in surr_names:
        base[name] = {
            "sparse": {
                "n_train_samples": np.array([100, 50, 25]),
                "model_errors": np.array([0.15, 0.2, 0.28]),
            }
        }
    return base


def make_metrics_for_batchsize(surr_names):
    base = {}
    for name in surr_names:
        base[name] = {
            "batch_size": {
                "batch_elements": np.array([32, 64, 128]),
                "model_errors": np.array([0.12, 0.11, 0.13]),
            }
        }
    return base


def make_metrics_for_UQ(surr_names, timesteps):
    base = {}
    T = len(timesteps)
    for name in surr_names:
        uq_std = np.full((1, T, 1), 0.25)
        uq_err = np.full((1, T, 1), 0.2)
        base[name] = {
            "timesteps": np.array(timesteps),
            "accuracy": {"absolute_errors_log": np.full((1, T, 1), 0.18)},
            "UQ": {
                "pred_uncertainty_log": uq_std,
                "absolute_errors_log": uq_err,
                "axis_max": 1,
                "max_counts": 1,
                "correlation_metrics_log": 0.4,
                "targets_log": np.zeros((1, T, 1)),
            },
        }
    return base


@pytest.fixture(autouse=True)
def stub_plots_and_io(monkeypatch):
    calls = []
    for fn in [
        # only stub out any pure-plot routines, but leave the CSV writers alone
        "inference_time_bar_plot",
        "plot_comparative_gradient_heatmaps",
        "plot_generalization_error_comparison",
        "plot_uncertainty_over_time_comparison",
        "plot_comparative_error_correlation_heatmaps",
        "plot_mean_deltadex_over_time_main_vs_ensemble",
        "plot_catastrophic_detection_curves",
        "plot_errors_over_time",
        "plot_error_distribution_comparative",
        "plot_loss_comparison",
        "plot_loss_comparison_equal",
        "plot_loss_comparison_train_duration",
        "plot_error_distribution_comparative",
    ]:
        monkeypatch.setattr(bf, fn, lambda *a, _n=fn, **k: calls.append((_n, a, k)))
    return calls


@pytest.fixture
def cfg():
    return {
        "training_id": "TID",
        "devices": ["cpu"],
        "losses": True,
        "gradients": True,
        "timing": True,
        "interpolation": {"enabled": True},
        "extrapolation": {"enabled": True},
        "sparse": {"enabled": True},
        "batch_scaling": {"enabled": True},
        "uncertainty": {"enabled": True},
        "epochs": [5],
        "surrogates": ["M1"],
        "relative_error_threshold": 0.0,
    }


def test_compare_main_losses(stub_plots_and_io, cfg, monkeypatch):
    metrics = make_metrics_for_main(["M1"])

    # stub get_surrogate -> our fake model class
    class Fake:
        def __init__(self, device, n_quantities, n_timesteps, n_parameters, config):
            self.train_loss = 0.11
            self.test_loss = 0.22
            self.train_duration = 3.14

        def load(self, *args, **kw):
            pass

    monkeypatch.setattr(bf, "get_surrogate", lambda n: Fake)
    monkeypatch.setattr(bf, "get_model_config", lambda *a, **k: {})
    bf.compare_main_losses(metrics, cfg)
    names = [c[0] for c in stub_plots_and_io]
    # expect the three plotting calls, in order
    assert names[:3] == [
        "plot_loss_comparison",
        "plot_loss_comparison_equal",
        "plot_loss_comparison_train_duration",
    ]


def test_compare_errors(stub_plots_and_io, cfg):
    timesteps = [0.0, 1.0, 2.0]
    metrics = make_metrics_for_relative(["M1"], timesteps)
    # also include deltadex branch to verify both paths
    metrics["M1"]["accuracy"]["absolute_errors_log"] = np.arange(
        len(timesteps)
    ).reshape(1, len(timesteps), 1)

    bf.compare_errors(metrics, cfg)
    # mean and median come from np.mean/median over rel errors
    mean_err = np.mean(metrics["M1"]["accuracy"]["relative_errors"], axis=(0, 2))
    median_err = np.median(metrics["M1"]["accuracy"]["relative_errors"], axis=(0, 2))
    # first call to plot_errors_over_time (relative)
    _n, args, kw = stub_plots_and_io[0]
    assert _n == "plot_errors_over_time"
    # args = ( mean_dict, median_dict, timesteps, cfg )
    assert pytest.approx(list(args[0].values())[0]) == mean_err
    assert pytest.approx(list(args[1].values())[0]) == median_err
    assert np.all(args[2] == timesteps)
    assert kw.get("mode") == "relative"
    # second call
    assert stub_plots_and_io[1][0] == "plot_error_distribution_comparative"
    assert stub_plots_and_io[1][2].get("mode") == "relative"

    # third and fourth calls should be for Δdex branch
    assert stub_plots_and_io[2][0] == "plot_errors_over_time"
    assert stub_plots_and_io[2][2].get("mode") == "deltadex"
    assert stub_plots_and_io[3][0] == "plot_error_distribution_comparative"
    assert stub_plots_and_io[3][2].get("mode") == "deltadex"


def test_compare_inference_time(stub_plots_and_io, cfg):
    metrics = make_metrics_for_timing(["M1"])
    bf.compare_inference_time(metrics, cfg)

    name, args, kw = stub_plots_and_io[0]
    assert name == "inference_time_bar_plot"

    # now args has 5 elements, show_title is in kw
    surrogates, means, stds, conf, save_flag = args
    assert surrogates == ["M1"]
    assert means == [1.23]
    assert stds == [0.45]
    assert conf is cfg
    assert save_flag is True
    assert kw.get("show_title", False) is True


def test_compare_gradients(stub_plots_and_io, cfg):
    m = make_metrics_for_dynamic(["M1"])
    bf.compare_gradients(m, cfg)

    name, args, kw = stub_plots_and_io[0]
    assert name == "plot_comparative_gradient_heatmaps"

    # now args has 7 elements, show_title is in kw
    grads, abs_errs, corrs, max_grads, max_errs, max_counts, conf = args
    assert list(grads.keys()) == ["M1"]
    assert corrs["M1"] == 0.5
    assert max_counts["M1"] == 8
    assert conf is cfg
    assert kw.get("show_title", False) is True


def test_compare_interpolation(stub_plots_and_io, cfg):
    m = make_metrics_for_interpolation(["M1", "M2"])
    bf.compare_interpolation(m, cfg)
    name, args, kw = stub_plots_and_io[0]
    assert name == "plot_generalization_error_comparison"
    surrogates, intervals, model_errors, xlabel, filename, conf = args
    assert surrogates == ["M1", "M2"]
    assert xlabel == "Interpolation Interval"
    assert filename == "errors_interpolation.png"
    assert all(isinstance(arr, np.ndarray) for arr in intervals)
    assert all(isinstance(arr, np.ndarray) for arr in model_errors)
    assert conf is cfg
    assert kw.get("show_title", False) is True


def test_compare_extrapolation(stub_plots_and_io, cfg):
    m = make_metrics_for_extrapolation(["M1"])
    bf.compare_extrapolation(m, cfg)
    name, args, kw = stub_plots_and_io[0]
    assert name == "plot_generalization_error_comparison"
    surrogates, cutoffs, model_errors, xlabel, filename, conf = args
    assert surrogates == ["M1"]
    assert xlabel == "Extrapolation Cutoff"
    assert filename == "errors_extrapolation.png"
    assert isinstance(cutoffs[0], np.ndarray)
    assert isinstance(model_errors[0], np.ndarray)
    assert conf is cfg


def test_compare_sparse(stub_plots_and_io, cfg):
    m = make_metrics_for_sparse(["M1"])
    bf.compare_sparse(m, cfg)
    name, args, kw = stub_plots_and_io[0]
    assert name == "plot_generalization_error_comparison"
    surrogates, n_train_samples, model_errors, xlabel, filename, conf = args
    assert surrogates == ["M1"]
    assert xlabel == "Number of Training Samples"
    assert filename == "errors_sparse.png"
    assert isinstance(n_train_samples[0], np.ndarray)
    assert isinstance(model_errors[0], np.ndarray)
    assert conf is cfg


def test_compare_batchsize(stub_plots_and_io, cfg):
    m = make_metrics_for_batchsize(["M1"])
    bf.compare_batchsize(m, cfg)
    name, args, kw = stub_plots_and_io[0]
    assert name == "plot_generalization_error_comparison"
    surrogates, batch_elements, model_errors, xlabel, filename, conf = args
    assert surrogates == ["M1"]
    assert xlabel == "Batch Size"
    assert filename == "errors_batch_size.png"
    assert isinstance(batch_elements[0], np.ndarray)
    assert isinstance(model_errors[0], np.ndarray)
    assert conf is cfg


def test_compare_UQ(stub_plots_and_io, cfg):
    timesteps = [0.0, 1.0, 2.0]
    m = make_metrics_for_UQ(["M1"], timesteps)
    bf.compare_UQ(m, cfg)
    names = [c[0] for c in stub_plots_and_io]
    assert names[:4] == [
        "plot_mean_deltadex_over_time_main_vs_ensemble",
        "plot_uncertainty_over_time_comparison",
        "plot_comparative_error_correlation_heatmaps",
        "plot_catastrophic_detection_curves",
    ]


def test_tabular_comparison_creates_files(tmp_path, stub_plots_and_io, monkeypatch):
    metrics = {
        "M1": {
            "accuracy": {
                "root_mean_squared_error_real": 0.1,
                "mean_absolute_error_real": 0.2,
                "median_absolute_error_real": 0.15,
                "percentile_absolute_error_real": 0.25,
                "root_mean_squared_error_log": 0.05,
                "mean_absolute_error_log": 0.04,
                "median_absolute_error_log": 0.035,
                "percentile_absolute_error_log": 0.06,
                "mean_relative_error": 0.3,
                "median_relative_error": 0.25,
                "percentile_relative_error": 0.35,
                "main_model_epochs": 4,
                "main_model_training_time": 7.0,
                "error_percentile": 99,
            }
        },
        "M2": {
            "accuracy": {
                "root_mean_squared_error_real": 0.08,
                "mean_absolute_error_real": 0.18,
                "median_absolute_error_real": 0.14,
                "percentile_absolute_error_real": 0.22,
                "root_mean_squared_error_log": 0.04,
                "mean_absolute_error_log": 0.03,
                "median_absolute_error_log": 0.028,
                "percentile_absolute_error_log": 0.05,
                "mean_relative_error": 0.25,
                "median_relative_error": 0.2,
                "percentile_relative_error": 0.3,
                "main_model_epochs": 5,
                "main_model_training_time": 3.0,
                "error_percentile": 99,
            }
        },
    }

    cfg = {
        "training_id": tmp_path.name,
        "timing": False,
        "gradients": False,
        "compute": False,
        "uncertainty": {"enabled": False},
        "interpolation": {"enabled": False},
        "extrapolation": {"enabled": False},
        "sparse": {"enabled": False},
        "batch_scaling": {"enabled": False},
        "verbose": False,
        "iterative": False,
    }

    # run inside tmp_path
    monkeypatch.chdir(tmp_path)
    # manually create results/<study_id>
    os.makedirs(tmp_path / "results" / cfg["training_id"], exist_ok=True)

    stub_plots_and_io.clear()
    bf.tabular_comparison(metrics, cfg)

    # ensure the text and CSV files on disk
    assert (tmp_path / "results" / cfg["training_id"] / "metrics_table.txt").exists()
    assert (tmp_path / "results" / cfg["training_id"] / "metrics_table.csv").exists()
    assert (tmp_path / "results" / cfg["training_id"] / "all_metrics.csv").exists()
