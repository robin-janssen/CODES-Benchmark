import numpy as np
import pytest
import os
import codes.benchmark.bench_fcts as bf


# Helpers to build dummy metrics
def make_metrics_for_main(surr_names, n_timesteps=4, n_quantities=3):
    # for compare_main_losses
    metrics = {}
    for i, name in enumerate(surr_names):
        metrics[name] = {
            "timesteps": np.zeros(n_timesteps),
            "accuracy": {"absolute_errors": np.zeros((1, n_timesteps, n_quantities))},
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
            "accuracy": {"absolute_errors": np.zeros((1, n_timesteps, n_quantities))},
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


@pytest.fixture(autouse=True)
def stub_plots_and_io(monkeypatch):
    calls = []
    for fn in [
        # only stub out any pure-plot routines, but leave the CSV writers alone
        "inference_time_bar_plot",
        "plot_comparative_dynamic_correlation_heatmaps",
        "plot_generalization_error_comparison",
        "plot_uncertainty_over_time_comparison",
        "plot_comparative_error_correlation_heatmaps",
        "plot_error_distribution_comparative",
        "plot_uncertainty_confidence",
        "plot_loss_comparison",
        "plot_loss_comparison_equal",
        "plot_loss_comparison_train_duration",
        "plot_relative_errors",
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


def test_compare_relative_errors(stub_plots_and_io, cfg):
    timesteps = [0.0, 1.0, 2.0]
    metrics = make_metrics_for_relative(["M1"], timesteps)
    bf.compare_relative_errors(metrics, cfg)
    # mean and median come from np.mean/median over rel errors
    mean_err = np.mean(metrics["M1"]["accuracy"]["relative_errors"], axis=(0, 2))
    median_err = np.median(metrics["M1"]["accuracy"]["relative_errors"], axis=(0, 2))
    # first call to plot_relative_errors
    _n, args, kw = stub_plots_and_io[0]
    assert _n == "plot_relative_errors"
    # args = ( mean_dict, median_dict, timesteps, cfg )
    assert pytest.approx(list(args[0].values())[0]) == mean_err
    assert pytest.approx(list(args[1].values())[0]) == median_err
    assert np.all(args[2] == timesteps)
    # second call
    assert stub_plots_and_io[1][0] == "plot_error_distribution_comparative"


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


def test_compare_dynamic_accuracy(stub_plots_and_io, cfg):
    m = make_metrics_for_dynamic(["M1"])
    bf.compare_dynamic_accuracy(m, cfg)

    name, args, kw = stub_plots_and_io[0]
    assert name == "plot_comparative_dynamic_correlation_heatmaps"

    # now args has 7 elements, show_title is in kw
    grads, abs_errs, corrs, max_grads, max_errs, max_counts, conf = args
    assert list(grads.keys()) == ["M1"]
    assert corrs["M1"] == 0.5
    assert max_counts["M1"] == 8
    assert conf is cfg
    assert kw.get("show_title", False) is True


def test_compare_UQ_and_confidence(stub_plots_and_io, cfg, monkeypatch):
    base = make_metrics_for_generalization(["M1"])
    # ADD a dummy timesteps array
    base["M1"]["timesteps"] = np.array([0.0, 1.0])

    # stub out plot_uncertainty_confidence to return known scores
    monkeypatch.setattr(bf, "plot_uncertainty_confidence", lambda *a, **k: {"M1": 0.42})

    bf.compare_UQ(base, cfg)

    # after compare_UQ, confidence_scores should exist in metrics
    assert base["M1"]["UQ"]["confidence_scores"] == 0.42


def test_tabular_comparison_creates_files(tmp_path, stub_plots_and_io):
    metrics = {
        "M1": {
            "accuracy": {
                "mean_squared_error": 0.1,
                "mean_absolute_error": 0.2,
                "mean_relative_error": 0.3,
                "main_model_epochs": 4,
                "main_model_training_time": 7.0,
            }
        },
        "M2": {
            "accuracy": {
                "mean_squared_error": 0.01,
                "mean_absolute_error": 0.02,
                "mean_relative_error": 0.03,
                "main_model_epochs": 5,
                "main_model_training_time": 3.0,
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
        "uncertainty": {"enabled": False},
        "verbose": False,
    }

    # run inside tmp_path
    os.chdir(tmp_path)
    # manually create results/<study_id>
    os.makedirs(tmp_path / "results" / cfg["training_id"], exist_ok=True)

    stub_plots_and_io.clear()
    bf.tabular_comparison(metrics, cfg)

    # ensure the text and CSV files on disk
    assert (tmp_path / "results" / cfg["training_id"] / "metrics_table.txt").exists()
    assert (tmp_path / "results" / cfg["training_id"] / "metrics_table.csv").exists()
    assert (tmp_path / "results" / cfg["training_id"] / "all_metrics.csv").exists()
