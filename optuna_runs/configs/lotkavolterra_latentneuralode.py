config = {
    "surrogate": {
        "name": "LatentNeuralODE",
    },
    "dataset": {
        "name": "lotka_volterra",
        "log10_transform": True,
        "normalise": "minmax",
        "subset_factor": 1,
    },
    "device": "cuda:3",
    "seed": 42,
    "batch_size": 128,
    "epochs": 5000,
    "n_trials": 200,
    "optuna_params": {
        "latent_features": {"type": "int", "low": 1, "high": 10},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "layers_factor": {"type": "int", "low": 1, "high": 100},
        "coder_activation": {
            "type": "categorical",
            "choices": ["ReLU", "LeakyReLU", "Tanh", "GELU", "Softplus"],
        },
        "ode_activation": {
            "type": "categorical",
            "choices": ["ReLU", "LeakyReLU", "Tanh", "GELU", "Softplus"],
        },
        "ode_tanh_reg": {"type": "categorical", "choices": [True, False]},
    },
}
