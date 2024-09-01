config = {
    "surrogate": {
        "name": "MultiONet",
    },
    "dataset": {
        "name": "osu2008",
        "log10_transform": True,
        "normalise": "minmax",
    },
    "device": "cuda:5",
    "seed": 42,
    "batch_size": 1024,
    "epochs": 400,
    "n_trials": 200,
    "optuna_params": {
        "branch_hidden_layers": {"type": "int", "low": 2, "high": 8},
        "trunk_hidden_layers": {"type": "int", "low": 2, "high": 8},
        "hidden_size": {"type": "int", "low": 50, "high": 200},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "output_factor": {"type": "int", "low": 1, "high": 30},
        "activation": {
            "type": "categorical",
            "choices": ["ReLU", "LeakyReLU", "Tanh", "GELU", "Softplus"],
        },
    },
}
