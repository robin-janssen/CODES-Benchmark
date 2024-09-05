config = {
    "surrogate": {
        "name": "LatentNeuralODE",
    },
    "dataset": {
        "name": "osu2008",
        "log10_transform": True,
        "normalise": "minmax",
    },
    "device": "cuda:1",
    "seed": 42,
    "batch_size": 1024,
    "epochs": 5000,
    "n_trials": 200,
    "optuna_params": {
        "latent_features": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "layers_factor": {"type": "int", "low": 1, "high": 50},
        "ode_activation": {
            "type": "categorical",
            "choices": ["ReLU", "LeakyReLU", "Tanh", "GELU", "Softplus"],
        },
        "ode_tanh_reg": {"type": "categorical", "choices": [True, False]},
    },
}
