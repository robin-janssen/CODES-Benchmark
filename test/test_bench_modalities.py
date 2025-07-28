import pytest
import numpy as np
import torch
from unittest.mock import patch

from codes.benchmark.bench_fcts import (
    evaluate_interpolation,
    evaluate_extrapolation,
    evaluate_sparse,
    evaluate_batchsize,
    evaluate_UQ,
)


# Dummy model to record load calls
class DummyModel:
    def __init__(self, device, n_quantities, n_timesteps, n_parameters, config):
        self._loads = []

    def load(self, training_id, surr_name, model_identifier):
        self._loads.append(model_identifier)

    def predict(self, data_loader):
        # targets always zero.  Shape (batch=2, timesteps=4, quantities=1).
        preds = torch.rand(2, 4, 1)
        targets = torch.rand(2, 4, 1)
        return preds, targets


# Two standalone fakes: one for heatmap (returns tuple), one for all others (returns None)
def _fake_heatmap(*args, **kwargs):
    return ([], [])


def _fake_noop(*args, **kwargs):
    return None


@pytest.fixture(autouse=True)
def patch_plots():
    import codes.benchmark.bench_fcts as bf

    fake_impl = {}
    for name in dir(bf):
        if not name.startswith("plot_"):
            continue
        if name == "plot_error_correlation_heatmap":
            fake_impl[name] = _fake_heatmap
        else:
            fake_impl[name] = _fake_noop

    with patch.multiple("codes.benchmark.bench_fcts", **fake_impl):
        yield


@pytest.mark.parametrize(
    "raw_vals, cfg_key, func, main_bs, expected_nums",
    [
        ([2, 3, 5], "interpolation", evaluate_interpolation, None, [1, 2, 3, 5]),
        ([1, 2, 4], "extrapolation", evaluate_extrapolation, None, [1, 2, 4]),
        ([2, 4, 8], "sparse", evaluate_sparse, None, [1, 2, 4, 8]),
        ([0.5, 2], "batch_scaling", evaluate_batchsize, 8, [4, 8, 16]),
        (3, "uncertainty", evaluate_UQ, None, [0, 1, 2]),
    ],
)
def test_modality_variations(raw_vals, cfg_key, func, main_bs, expected_nums):
    surr = "TestSurr"
    cfg = {"training_id": "TID", "surrogates": [surr]}
    if cfg_key == "uncertainty":
        cfg["uncertainty"] = {"enabled": True, "ensemble_size": raw_vals}
    else:
        cfg[cfg_key] = {"enabled": True}
        subkey = {
            "interpolation": "intervals",
            "extrapolation": "cutoffs",
            "sparse": "factors",
            "batch_scaling": "sizes",
        }[cfg_key]
        cfg[cfg_key][subkey] = raw_vals
    if cfg_key == "batch_scaling":
        cfg["batch_size"] = [main_bs]

    timesteps = np.arange(4)
    loader = object()
    labels = ["q"] if func is evaluate_interpolation else None

    model = DummyModel(None, 1, len(timesteps), 0, {})

    # invoke
    if func is evaluate_interpolation:
        metrics = func(model, surr, loader, timesteps, cfg, labels)
    elif func is evaluate_extrapolation:
        metrics = func(model, surr, loader, timesteps, cfg, labels)
    elif func is evaluate_sparse:
        metrics = func(model, surr, loader, timesteps, n_train_samples=10, conf=cfg)
    elif func is evaluate_batchsize:
        metrics = func(model, surr, loader, timesteps, cfg)
    else:
        metrics = func(model, surr, loader, timesteps, cfg, labels=None)

    lower = surr.lower()
    # build expected identifiers
    ids = []
    if cfg_key == "interpolation":
        for i in expected_nums:
            ids.append(f"{lower}_main" if i == 1 else f"{lower}_interpolation_{i}")
    elif cfg_key == "extrapolation":
        max_c = len(timesteps)
        for c in expected_nums:
            ids.append(f"{lower}_main" if c == max_c else f"{lower}_extrapolation_{c}")
    elif cfg_key == "sparse":
        for f in expected_nums:
            ids.append(f"{lower}_main" if f == 1 else f"{lower}_sparse_{f}")
    elif cfg_key == "batch_scaling":
        for bs in expected_nums:
            ids.append(f"{lower}_main" if bs == main_bs else f"{lower}_batchsize_{bs}")
    else:  # uncertainty
        for idx in expected_nums:
            ids.append(f"{lower}_main" if idx == 0 else f"{lower}_UQ_{idx}")

    assert model._loads == ids

    prefix = {
        "interpolation": "interval",
        "extrapolation": "cutoff",
        "sparse": "factor",
        "batch_scaling": "batch_size",
        "uncertainty": None,
    }[cfg_key]

    if cfg_key != "uncertainty":
        for num in expected_nums:
            assert f"{prefix} {num}" in metrics
    else:
        assert "average_uncertainty" in metrics
