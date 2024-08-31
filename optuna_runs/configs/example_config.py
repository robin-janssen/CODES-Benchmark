config = {
    "surrogate": {
        "name": "MultiONet",
    },
    "dataset": {
        "name": "osu2008",
        "log10_transform": False,
        "normalise": "minmax",
    },
    "device": "cuda:0",
    "seed": 42,
    "batch_size": 1024,
    "epochs": 500,
    "study_name": "multionet_osu",
    "optuna_params": {
        "branch_hidden_layers": {"type": "int", "low": 2, "high": 8},
        "trunk_hidden_layers": {"type": "int", "low": 2, "high": 8},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "output_factor": {"type": "int", "low": 1, "high": 30},
        "activation_function": {
            "type": "categorical",
            "choices": ["relu", "tanh", "sigmoid"],
        },
    },
}
