import torch
import optuna

from surrogates.LatentPolynomial.latent_poly import LatentPoly
from surrogates.LatentPolynomial.latent_poly_config import LatentPolynomialConfigOSU
from data import check_and_load_data

DEVICE = "cuda:0"
EPOCHS = 2000


def objective(trial):

    config = LatentPolynomialConfigOSU()

    activations = ["relu", "tanh", "leaky_relu"]
    layers = ([32, 16, 8], [64, 32, 16], [64, 32, 16, 8], [32, 8])

    coder_activation = trial.suggest_categorical(
        "coder_activation",
        choices=activations,
    )

    match coder_activation:
        case "relu":
            coder_activation = torch.nn.ReLU()
        case "tanh":
            coder_activation = torch.nn.Tanh()
        case "leaky_relu":
            coder_activation = torch.nn.LeakyReLU()

    coder_layers = trial.suggest_categorical(
        "coder_layers",
        choices=layers,
    )

    degree = trial.suggest_int("degree", 1, 5)

    config.coder_activation = coder_activation
    config.coder_layers = coder_layers
    config.coder_hidden = len(coder_layers) + 1
    config.degree = degree
    config.device = DEVICE
    config.epochs = EPOCHS

    model = LatentPoly(config=config, device=DEVICE)

    train_data, test_data, _, timesteps, _, data_params = check_and_load_data(
        dataset_name="osu2008",
        verbose=False,
        log=False,
        normalisation_mode="minmax",
    )

    train_loader, test_loader, _ = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=None,
        timesteps=timesteps,
        batch_size=128,
        shuffle=True,
    )

    model.fit(train_loader, test_loader, timesteps, EPOCHS)

    return model.test_loss[-20:].mean()


# optuna.delete_study(
#     study_name="LatentPoly Optimization",
#     storage="sqlite:////export/data/isulzer/DON-vs-NODE/study/end_to_end.db",
# )

study = optuna.create_study(
    direction="minimize",
    study_name="LatentPoly Optimization",
    storage="sqlite:////export/data/isulzer/DON-vs-NODE/study/end_to_end.db",
)
study.optimize(objective, n_trials=100)
print(study.best_params)
