config = {
    "surrogate": {
        "name": "FullyConnected",
    },
    "dataset": {
        "name": "lotka_volterra",
        "log10_transform": True,
        "normalise": "minmax",
        "subset_factor": 1,
    },
    "device": "cuda:1",
    "seed": 42,
    "batch_size": 1024,
    "epochs": 500,
    "n_trials": 200,
    "optuna_params": {
        "hidden_size": {"type": "int", "low": 20, "high": 500},
        "num_hidden_layers": {"type": "int", "low": 1, "high": 10},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "activation": {
            "type": "categorical",
            "choices": ["ReLU", "LeakyReLU", "Tanh", "GELU", "Softplus"],
        },
    },
}
