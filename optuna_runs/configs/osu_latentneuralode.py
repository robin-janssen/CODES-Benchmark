config = {
    "surrogate": {
        "name": "FullyConnected",
    },
    "dataset": {
        "name": "osu2008",
        "log10_transform": True,
        "normalise": "minmax",
    },
    "device": "cuda:6",
    "seed": 42,
    "batch_size": 1024,
    "epochs": 400,
    "study_name": "fullyconnected_osu",
    "optuna_params": {
        "hidden_size": {"type": "int", "low": 50, "high": 300},
        "num_hidden_layers": {"type": "int", "low": 2, "high": 8},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    },
}
