config = {
    "surrogate": {
        "name": "FullyConnected",
    },
    "dataset": {
        "name": "branca24",
        "log10_transform": True,
        "normalise": "minmax",
        "subset_factor": 8,
    },
    "device": "cuda:7",
    "seed": 42,
    "batch_size": 1024,
    "epochs": 200,
    "n_trials": 200,
    "optuna_params": {
        "hidden_size": {"type": "int", "low": 50, "high": 300},
        "num_hidden_layers": {"type": "int", "low": 2, "high": 8},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    },
}
