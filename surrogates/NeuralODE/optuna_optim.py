import sys
sys.path.append("/export/data/isulzer/DON-vs-NODE/")

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import Tensor
import numpy as np
import optuna



from surrogates.NeuralODE.neural_ode_config import NeuralODEConfigOSU as Config
from surrogates.NeuralODE.neural_ode import NeuralODE
from data import check_and_load_data
from utils import time_execution


class NeuralODESub(NeuralODE):

    @time_execution
    def fit(
        self,
        trial,
        train_loader: DataLoader | Tensor,
        test_loader: DataLoader | Tensor,
        timesteps: np.ndarray | Tensor,
        epochs: int | None,
        position: int = 0,
        description: str = "Training NeuralODE",
    ) -> None:
        """
        Fits the model to the training data. Sets the train_loss and test_loss attributes.

        Args:
            train_loader (DataLoader): The data loader for the training data.
            test_loader (DataLoader): The data loader for the test data.
            timesteps (np.ndarray | Tensor): The array of timesteps.
            epochs (int | None): The number of epochs to train the model. If None, uses the value from the config.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.

        Returns:
            None
        """
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps).to(self.config.device)
        epochs = self.config.epochs if epochs is None else epochs

        # TODO: make Optimizer and scheduler configable
        optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

        scheduler = None
        if self.config.final_learning_rate is not None:
            scheduler = CosineAnnealingLR(
                optimizer, self.config.epochs, eta_min=self.config.final_learning_rate
            )

        losses = torch.empty((epochs, len(train_loader)))
        test_losses = torch.empty((epochs))
        accuracy = torch.empty((epochs))

        progress_bar = self.setup_progress_bar(epochs, position, description)

        for epoch in progress_bar:
            for i, x_true in enumerate(train_loader):
                optimizer.zero_grad()
                # x0 = x_true[:, 0, :]
                x_pred = self.model.forward(x_true, timesteps)
                loss = self.model.total_loss(x_true, x_pred)
                loss.backward()
                optimizer.step()
                losses[epoch, i] = loss.item()

                # TODO: make configable
                if epoch == 10 and i == 0:
                    with torch.no_grad():
                        self.model.renormalize_loss_weights(x_true, x_pred)

            clr = optimizer.param_groups[0]["lr"]
            print_loss = f"{losses[epoch, -1].item():.2e}"
            progress_bar.set_postfix({"loss": print_loss, "lr": f"{clr:.1e}"})

            if scheduler is not None:
                scheduler.step()

            with torch.inference_mode():
                self.model.eval()
                preds, targets = self.predict(test_loader, timesteps)
                self.model.train()
                loss = self.model.l2_loss(preds, targets)
                test_losses[epoch] = loss
                accuracy[epoch] = 1.0 - torch.mean(
                    torch.abs(preds - targets) / torch.abs(targets)
                )
            if trial.should_prune():
                raise optuna.TrialPruned()

        progress_bar.close()

        self.train_loss = torch.mean(losses, dim=1)
        self.test_loss = test_losses
        self.accuracy = accuracy


DEVICE = "cuda:2"
EPOCHS = 1000


def objective(trial):

    config = Config()

    choices = ["relu", "tanh", "leaky_relu"]

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    coder_activation = trial.suggest_categorical(
        "coder_activation",
        choices=choices,
    )
    ode_activation = trial.suggest_categorical(
        "ode_activation",
        choices=choices,
    )

    match coder_activation:
        case "relu":
            coder_activation = torch.nn.ReLU()
        case "tanh":
            coder_activation = torch.nn.Tanh()
        case "leaky_relu":
            coder_activation = torch.nn.LeakyReLU()

    match ode_activation:
        case "relu":
            ode_activation = torch.nn.ReLU()
        case "tanh":
            ode_activation = torch.nn.Tanh()
        case "leaky_relu":
            ode_activation = torch.nn.LeakyReLU()

    ode_layer_width = trial.suggest_categorical("ode_layer_width", (32, 64, 128, 256))
    ode_depth = trial.suggest_categorical("ode_depth", (3, 4, 5, 6))

    config.learning_rate = learning_rate
    config.coder_activation = coder_activation
    config.ode_activation = ode_activation
    config.ode_layer_width = ode_layer_width
    config.ode_hidden = ode_depth

    model = NeuralODESub(device=DEVICE)

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
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.fit(trial, train_loader, test_loader, timesteps, EPOCHS)

    return model.test_loss[-20:].mean()


study = optuna.create_study(
    direction="minimize",
    study_name="NeuralODE Optimization",
    pruner=optuna.pruners.PatientPruner(
        optuna.pruners.PercentilePruner(percentile=0.7), patience=10
    ),
    # storage="sqlite:///end_to_end.db",
)
study.optimize(objective, n_trials=100)
print(study.best_params)
