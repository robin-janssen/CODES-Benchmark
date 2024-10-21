config = {
    "surrogate": {
        "name": "MultiONet",
    },
    "dataset": {
        "name": "simple_ode",
        "log10_transform": True,
        "normalise": "minmax",
        "subset_factor": 1,
    },
    "device": "cuda:3",
    "seed": 42,
    "batch_size": 1024,
    "epochs": 500,
    "n_trials": 200,
    "optuna_params": {
        "branch_hidden_layers": {"type": "int", "low": 1, "high": 10},
        "trunk_hidden_layers": {"type": "int", "low": 1, "high": 10},
        "hidden_size": {"type": "int", "low": 10, "high": 400},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "output_factor": {"type": "int", "low": 1, "high": 100},
        "activation": {
            "type": "categorical",
            "choices": ["ReLU", "LeakyReLU", "Tanh", "GELU", "Softplus"],
        },
    },
}
