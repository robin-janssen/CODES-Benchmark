config = {
    "surrogate": {
        "name": "LatentPoly",
    },
    "dataset": {
        "name": "branca24",
        "log10_transform": True,
        "normalise": "minmax",
        "subset_factor": 8,
    },
    "device": "cuda:6",
    "seed": 42,
    "batch_size": 128,
    "epochs": 2000,
    "n_trials": 200,
    "optuna_params": {
        "degree": {"type": "int", "low": 1, "high": 8},
        "latent_features": {"type": "int", "low": 3, "high": 10},
        "coder_activation": {
            "type": "categorical",
            "choices": ["ReLU", "LeakyReLU", "Tanh", "GELU", "Softplus"],
        },
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "layers_factor": {"type": "int", "low": 1, "high": 50},
    },
}
