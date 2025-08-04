import math
import queue
from datetime import datetime, timedelta

import pytest
from optuna.trial import TrialState

from codes.tune.optuna_fcts import (
    MODULE_REGISTRY,
    create_objective,
    make_optuna_params,
    maybe_set_runtime_threshold,
)


class DummyTrial:
    def __init__(self):
        self.suggested = {}

    def suggest_int(self, name, low, high, step=1):
        # always return low
        self.suggested[name] = low
        return low

    def suggest_float(self, name, low, high, log=False):
        # always return high
        self.suggested[name] = high
        return high

    def suggest_categorical(self, name, choices):
        # return first choice
        val = choices[0]
        self.suggested[name] = val
        return val


class DummyModel:
    def __init__(
        self, device, n_quantities, n_timesteps, n_parameters, config=None, **kwargs
    ):
        pass

    def to(self, device):
        pass

    def prepare_data(self, **kwargs):
        return "train_loader", "test_loader", None

    def fit(self, **kwargs):
        pass

    def predict(self, loader, leave_log=False):
        import torch

        t = torch.zeros((2, 4, 1))
        return t, t

    def save(self, **kwargs):
        pass


@pytest.fixture
def basic_params():
    return {
        "batch_size": {"type": "int", "low": 10, "high": 20, "step": 5},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.01, "log": True},
        "activation": {"choices": ["relu", "tanh", "identity"]},
        "loss_function": {"choices": ["mse", "smoothl1"]},
    }


@pytest.fixture
def conditional_params(basic_params):
    p = basic_params.copy()
    p["scheduler"] = {"choices": ["poly", "cosine"]}
    p["poly_power"] = {"type": "int", "low": 1, "high": 3}
    p["eta_min"] = {"type": "float", "low": 0.0, "high": 0.1}
    return p


def test_make_optuna_params_basic(basic_params):
    trial = DummyTrial()
    # only basic choices
    out = make_optuna_params(
        trial,
        {
            "batch_size": basic_params["batch_size"],
            "learning_rate": basic_params["learning_rate"],
            "activation": basic_params["activation"],
            "loss_function": basic_params["loss_function"],
        },
    )
    # int param should equal low
    assert out["batch_size"] == basic_params["batch_size"]["low"]
    # float param should equal high
    assert math.isclose(out["learning_rate"], basic_params["learning_rate"]["high"])
    # activation returns first choice
    expected_cls = MODULE_REGISTRY[basic_params["activation"]["choices"][0]]
    assert isinstance(out["activation"], expected_cls)
    # loss_function returns first choice
    expected_cls = MODULE_REGISTRY[basic_params["loss_function"]["choices"][0]]
    assert isinstance(out["loss_function"], expected_cls)


def test_make_optuna_params_conditional(conditional_params):
    trial = DummyTrial()
    # include scheduler to trigger poly branch
    params = conditional_params.copy()
    params["scheduler"]["choices"] = ["poly"]
    out = make_optuna_params(trial, params)
    # since scheduler chosen 'poly', poly_power must be suggested
    assert "poly_power" in out
    # cos branch
    trial2 = DummyTrial()
    p2 = conditional_params.copy()
    p2["scheduler"]["choices"] = ["cosine"]
    out2 = make_optuna_params(trial2, p2)
    assert "eta_min" in out2


# --------- Tests for maybe_set_runtime_threshold ----------
class FakeTrial:
    def __init__(self, num, state, start, complete=None):
        self.number = num
        self.state = state
        self.datetime_start = start
        self.datetime_complete = complete


class FakeStudy:
    def __init__(self, trials):
        self._trials = trials
        self.user_attrs = {}

    def get_trials(self, deepcopy=False):
        return self._trials

    def set_user_attr(self, key, val):
        self.user_attrs[key] = val


def test_maybe_set_runtime_threshold_not_enough():
    # only 1 complete trial, warmup_target=2
    t1 = FakeTrial(
        0,
        TrialState.COMPLETE,
        datetime.utcnow() - timedelta(seconds=5),
        datetime.utcnow(),
    )
    study = FakeStudy([t1])
    maybe_set_runtime_threshold(study, warmup_target=2)
    assert "runtime_threshold" not in study.user_attrs


def test_maybe_set_runtime_threshold_enough():
    now = datetime.utcnow()
    trials = []
    for i in range(3):
        trials.append(
            FakeTrial(
                i,
                TrialState.COMPLETE,
                now - timedelta(seconds=10 + i),
                now - timedelta(seconds=i),
            )
        )
    study = FakeStudy(trials)
    maybe_set_runtime_threshold(study, warmup_target=3)
    # after enough trials, attrs should be set
    assert "runtime_threshold" in study.user_attrs
    assert "warmup_mean" in study.user_attrs
    assert study.user_attrs["warmup_target"] == 3


# --------- Tests for create_objective ----------


def test_training_run_single_objective(monkeypatch, tmp_path):
    # Prepare dummy config
    config = {
        "dataset": {
            "name": "ds",
            "log10_transform": False,
            "normalise": "none",
            "tolerance": 1e-3,
        },
        "seed": 123,
        "surrogate": {"name": "Surr"},
        "optuna_params": {},
        "batch_size": 16,
        "epochs": 5,
        "target_percentile": 0.5,
        "multi_objective": False,
    }
    # Fake download
    monkeypatch.setattr("codes.tune.optuna_fcts.download_data", lambda *args, **k: None)
    # Fake data loaders
    import numpy as np

    dummy_data = np.zeros((2, 4, 1))
    dummy_params = np.zeros((2, 3))
    dummy_timesteps = np.arange(4)
    dummy_info = {}
    monkeypatch.setattr(
        "codes.tune.optuna_fcts.check_and_load_data",
        lambda *args, **kw: (
            (dummy_data, dummy_data, None),
            (dummy_params, dummy_params, None),
            dummy_timesteps,
            None,
            dummy_info,
            None,
        ),
    )
    monkeypatch.setattr("codes.tune.optuna_fcts.set_random_seeds", lambda *a, **k: None)

    monkeypatch.setattr("codes.tune.optuna_fcts.get_surrogate", lambda name: DummyModel)
    monkeypatch.setattr("codes.tune.optuna_fcts.get_model_config", lambda name, cfg: {})
    monkeypatch.setattr(
        "codes.tune.optuna_fcts.make_optuna_params", lambda trial, params: {}
    )
    # patch quantile
    import torch

    monkeypatch.setattr(torch, "quantile", lambda x, q: torch.tensor(7.0))
    # Run
    trial = type("T", (object,), {"number": 0})()
    val = __import__("codes.tune.optuna_fcts", fromlist=["training_run"]).training_run(
        trial, "cpu", 0, config, "study1"
    )
    assert isinstance(val, float) and val == 7.0


def test_training_run_multi_objective(monkeypatch, tmp_path):
    config = {
        "dataset": {
            "name": "ds",
            "log10_transform": False,
            "normalise": "none",
            "tolerance": 1e-3,
        },
        "seed": 456,
        "surrogate": {"name": "Surr"},
        "optuna_params": {},
        "batch_size": 8,
        "epochs": 5,
        "target_percentile": 0.5,
        "multi_objective": True,
    }
    # stub all as above
    monkeypatch.setattr("codes.tune.optuna_fcts.download_data", lambda *args, **k: None)
    import numpy as np

    dummy_data = np.ones((3, 5, 1))
    dummy_params = np.ones((3, 2))
    dummy_timesteps = np.arange(5)
    dummy_info = {}
    monkeypatch.setattr(
        "codes.tune.optuna_fcts.check_and_load_data",
        lambda *args, **kw: (
            (dummy_data, dummy_data, None),
            (dummy_params, dummy_params, None),
            dummy_timesteps,
            None,
            dummy_info,
            None,
        ),
    )
    monkeypatch.setattr("codes.tune.optuna_fcts.set_random_seeds", lambda *a, **k: None)

    class DummyModel2(DummyModel):
        def predict(self, loader, leave_log=False):
            import torch

            return torch.zeros((3, 5, 1)), torch.ones((3, 5, 1))

    monkeypatch.setattr(
        "codes.tune.optuna_fcts.get_surrogate", lambda name: DummyModel2
    )
    monkeypatch.setattr("codes.tune.optuna_fcts.get_model_config", lambda name, cfg: {})
    monkeypatch.setattr(
        "codes.tune.optuna_fcts.make_optuna_params", lambda trial, params: {}
    )
    monkeypatch.setattr(
        "codes.tune.optuna_fcts.measure_inference_time", lambda m, dur: [1.0, 2.0, 3.0]
    )
    import torch

    monkeypatch.setattr(torch, "quantile", lambda x, q: torch.tensor(5.0))

    trial = type("T", (object,), {"number": 2})()
    val = __import__("codes.tune.optuna_fcts", fromlist=["training_run"]).training_run(
        trial, "cpu", 1, config, "study_2"
    )
    # expect (loss, mean_inference)
    assert isinstance(val, tuple) and val[0] == 5.0 and val[1] == pytest.approx(2.0)


def test_create_objective_simple(monkeypatch):
    # stub training_run
    called = {}

    def fake_run(trial, device, slot, config, name):
        called["args"] = (device, slot, config, name)
        return 42.0

    monkeypatch.setattr("codes.tune.optuna_fcts.training_run", fake_run)

    device_queue = queue.Queue()
    device_queue.put(("cpu", 0))
    config = {"dataset": {"name": "ds"}}
    obj = create_objective(config, "study1", device_queue)

    class DummyT:
        number = 5

    trial = DummyT()
    result = obj(trial)
    # training_run returned 42.0
    assert result == 42.0
    # device put back
    assert not device_queue.empty()
    dev, slot = device_queue.get()
    assert dev == "cpu" and slot == 0
