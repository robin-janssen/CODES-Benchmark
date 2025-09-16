import pytest

from codes.benchmark import bench_fcts


@pytest.fixture(autouse=True)
def record_calls(monkeypatch):
    """
    Stub out all compare_* and plot_* functions so that calls
    just record their names into a shared list, instead of doing any real work.
    """
    calls = []
    names = [
        "compare_errors",
        "compare_main_losses",
        "compare_gradients",
        "compare_inference_time",
        "compare_interpolation",
        "compare_extrapolation",
        "compare_sparse",
        "plot_all_generalization_errors",
        "compare_batchsize",
        "compare_UQ",
        "tabular_comparison",
    ]
    for name in names:
        monkeypatch.setattr(
            bench_fcts,
            name,
            lambda *args, _n=name, **kw: calls.append(_n),
        )
    return calls


def make_dummy_metrics():
    """
    Build a minimal metrics dict that contains the keys
    your compare_models dispatcher will look up.
    Values themselves are never inspected by our stubs.
    """
    return {
        "M1": {
            "accuracy": {"relative_errors": None},
            "timesteps": None,
            "n_params": 0,
            # for each enabled branch add a dummy sub-dict:
            "timing": {
                "mean_inference_time_per_run": 1.0,
                "std_inference_time_per_run": 0.1,
            },
            "gradients": {
                "gradients": None,
                "avg_correlation": 0.0,
                "max_gradient": 0,
                "max_error": 0,
                "max_counts": 0,
            },
            "interpolation": {"intervals": [1], "model_errors": [0]},
            "extrapolation": {"cutoffs": [1], "model_errors": [0]},
            "sparse": {"n_train_samples": [10], "model_errors": [0]},
            "batch_size": {"batch_sizes": [32], "model_errors": [0]},
            "UQ": {
                "pred_uncertainty": None,
                "absolute_errors": None,
                "relative_errors": None,
                "axis_max": None,
                "max_counts": None,
                "correlation_metrics": None,
                "weighted_diff": None,
            },
        }
    }


@pytest.mark.parametrize(
    "flags, expected_sequence",
    [
        # all branches on
        (
            {
                "losses": True,
                "gradients": True,
                "timing": True,
                "interpolation": {"enabled": True},
                "extrapolation": {"enabled": True},
                "sparse": {"enabled": True},
                "batch_scaling": {"enabled": True},
                "uncertainty": {"enabled": True},
            },
            [
                "compare_errors",
                "compare_main_losses",
                "compare_gradients",
                "compare_inference_time",
                "compare_interpolation",
                "compare_extrapolation",
                "compare_sparse",
                "plot_all_generalization_errors",  # only if int+ext+sparse all enabled
                "compare_batchsize",
                "compare_UQ",
                "tabular_comparison",
            ],
        ),
        # only the mandatory relative-errors + table
        (
            {
                "losses": False,
                "gradients": False,
                "timing": False,
                "interpolation": {"enabled": False},
                "extrapolation": {"enabled": False},
                "sparse": {"enabled": False},
                "batch_scaling": {"enabled": False},
                "uncertainty": {"enabled": False},
            },
            [
                "compare_errors",
                "tabular_comparison",
            ],
        ),
        # losses but nothing else
        (
            {
                "losses": True,
                "gradients": False,
                "timing": False,
                "interpolation": {"enabled": False},
                "extrapolation": {"enabled": False},
                "sparse": {"enabled": False},
                "batch_scaling": {"enabled": False},
                "uncertainty": {"enabled": False},
            },
            [
                "compare_errors",
                "compare_main_losses",
                "tabular_comparison",
            ],
        ),
    ],
)
def test_compare_models_branching(record_calls, flags, expected_sequence):
    cfg = {
        "training_id": "test",
        "devices": ["cpu"],  # for compare_main_losses
        "losses": flags["losses"],
        "gradients": flags["gradients"],
        "timing": flags["timing"],
        "interpolation": flags["interpolation"],
        "extrapolation": flags["extrapolation"],
        "sparse": flags["sparse"],
        "batch_scaling": flags["batch_scaling"],
        "uncertainty": flags["uncertainty"],
    }
    metrics = make_dummy_metrics()

    bench_fcts.compare_models(metrics, cfg)

    assert record_calls == expected_sequence
