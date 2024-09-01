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
    "n_trials": 200,
    "optuna_params": {
        "degree": {"type": "int", "low": 1, "high": 8},
        "latent_features": {"type": "int", "low": 3, "high": 10},
        "activation": {
            "type": "categorical",
            "choices": ["ReLU", "LeakyReLU", "Tanh", "GELU", "Softplus"],
        },
    },
}
