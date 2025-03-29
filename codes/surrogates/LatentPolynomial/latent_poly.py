import numpy as np
import optuna
import torch
from schedulefree import AdamWScheduleFree
from torch import nn

# from torch.optim import Adam
from torch.utils.data import DataLoader

from codes.surrogates.AbstractSurrogate.surrogates import AbstractSurrogateModel
from codes.surrogates.LatentNeuralODE.latent_neural_ode import Decoder, Encoder
from codes.surrogates.LatentNeuralODE.utilities import ChemDataset
from codes.surrogates.LatentPolynomial.latent_poly_config import (
    LatentPolynomialBaseConfig,
)
from codes.utils import time_execution, worker_init_fn


class LatentPoly(AbstractSurrogateModel):
    """
    LatentPoly class for training a polynomial model on latent space trajectories.
    Includes an Encoder, Decoder and learnable Polynomial.

    Attributes:
        config (LatentPolynomialBaseConfig): The configuration for the model.
        model (PolynomialModelWrapper): The model for the polynomial.
        device (str): The device to use for training.

    Methods:
        forward: Perform a forward pass through the model.
        prepare_data: Prepares the data for training by creating a DataLoader object.
        fit: Fits the model to the training data. Sets the train_loss and test_loss attributes.
    """

    def __init__(
        self,
        device: str | None = None,
        n_chemicals: int = 29,
        n_timesteps: int = 100,
        model_config: dict | None = None,
    ):
        super().__init__(
            device=device,
            n_chemicals=n_chemicals,
            n_timesteps=n_timesteps,
            config=model_config,
        )
        self.config = LatentPolynomialBaseConfig(**self.config)
        coder_layers = [4, 2, 1]
        self.config.coder_layers = [
            layer * self.config.layers_factor for layer in coder_layers
        ]
        self.config.in_features = n_chemicals
        self.model = PolynomialModelWrapper(config=self.config, device=self.device)

    def forward(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the model.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predictions and targets
        """
        targets, timesteps = inputs[0], inputs[1]
        return self.model(targets, timesteps), targets

    def prepare_data(
        self,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray | None,
        dataset_val: np.ndarray | None,
        timesteps: np.ndarray,
        batch_size: int = 128,
        shuffle: bool = True,
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        """
        Prepares the data for training by creating DataLoader objects.

        Args:
            dataset_train (np.ndarray): The training dataset.
            dataset_test (np.ndarray): The test dataset.
            dataset_val (np.ndarray): The validation dataset.
            timesteps (np.ndarray): The array of timesteps.
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the training data.

        Returns:
            tuple[DataLoader, DataLoader | None, DataLoader | None]:
                - DataLoader for training data.
                - DataLoader for test data (None if no test data provided).
                - DataLoader for validation data (None if no validation data provided).
        """
        # Shuffle training data if required
        if shuffle:
            shuffled_indices = np.random.permutation(len(dataset_train))
            dataset_train = dataset_train[shuffled_indices]

        # Create training DataLoader
        dset_train = ChemDataset(dataset_train, timesteps, device=self.device)
        dataloader_train = DataLoader(
            dset_train,
            batch_size=batch_size,
            shuffle=False,  # Shuffle already handled manually above
            worker_init_fn=worker_init_fn,
            collate_fn=lambda x: (x[0], x[1]),
        )

        # Create test DataLoader (no shuffling)
        dataloader_test = None
        if dataset_test is not None:
            dset_test = ChemDataset(dataset_test, timesteps, device=self.device)
            dataloader_test = DataLoader(
                dset_test,
                batch_size=batch_size,
                shuffle=False,
                worker_init_fn=worker_init_fn,
                collate_fn=lambda x: (x[0], x[1]),
            )

        # Create validation DataLoader (no shuffling)
        dataloader_val = None
        if dataset_val is not None:
            dset_val = ChemDataset(dataset_val, timesteps, device=self.device)
            dataloader_val = DataLoader(
                dset_val,
                batch_size=batch_size,
                shuffle=False,
                worker_init_fn=worker_init_fn,
                collate_fn=lambda x: (x[0], x[1]),
            )

        return dataloader_train, dataloader_test, dataloader_val

    @time_execution
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        position: int = 0,
        description: str = "Training LatentPoly",
        multi_objective: bool = False,
    ) -> None:
        """
        Fits the model to the training data. Sets the train_loss and test_loss attributes.
        After 10 epochs, the loss weights are renormalized to scale the individual loss terms.

        Args:
            train_loader (DataLoader): The data loader for the training data.
            test_loader (DataLoader): The data loader for the test data.
            epochs (int | None): The number of epochs to train the model. If None, uses the value from the config.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.
            multi_objective (bool): Whether multi-objective optimization is used.
                                    If True, trial.report is not used (not supported by Optuna).
        """
        # optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        optimizer = AdamWScheduleFree(
            self.model.parameters(), lr=self.config.learning_rate
        )
        optimizer.train()

        losses = torch.empty((epochs, len(train_loader)))
        test_losses = torch.empty((epochs))
        MAEs = torch.empty((epochs))
        criterion = nn.MSELoss()

        progress_bar = self.setup_progress_bar(epochs, position, description)

        for epoch in progress_bar:

            for i, (x_true, timesteps) in enumerate(train_loader):
                optimizer.zero_grad()
                x_pred = self.model.forward(x_true, timesteps)
                loss = self.model.total_loss(x_true, x_pred)
                loss.backward()
                optimizer.step()
                losses[epoch, i] = loss.item()

                if epoch == 10 and i == 0:
                    with torch.no_grad():
                        self.model.renormalize_loss_weights(x_true, x_pred)

            clr = optimizer.param_groups[0]["lr"]
            # print_loss = f"{losses[epoch, -1].item():.2e}"
            # progress_bar.set_postfix({"loss": print_loss, "lr": f"{clr:.1e}"})

            with torch.inference_mode():
                self.model.eval()
                optimizer.eval()
                preds, targets = self.predict(test_loader)
                self.model.train()
                optimizer.train()
                # loss = self.model.total_loss(preds, targets)
                loss = criterion(preds, targets)
                test_losses[epoch] = loss
                MAEs[epoch] = self.L1(preds, targets).item()

                print_loss = f"{test_losses[epoch].item():.2e}"
                progress_bar.set_postfix({"loss": print_loss, "lr": f"{clr:.1e}"})

                if self.optuna_trial is not None and not multi_objective:
                    if epoch % self.trial_update_epochs == 0:
                        self.optuna_trial.report(loss, epoch)
                        if self.optuna_trial.should_prune():
                            raise optuna.TrialPruned()

        progress_bar.close()

        self.n_epochs = epoch
        self.train_loss = torch.mean(losses, dim=1)
        self.test_loss = test_losses
        self.MAE = MAEs


class PolynomialModelWrapper(nn.Module):
    """
    Wraps the Encoder, Decoder and Polynomial classes into a single model.

    Attributes:
        config (LatentPolynomialBaseConfig): The configuration for the model.
        loss_weights (list[float]): The loss weights for the model.
        device (str): The device to use for training.
        encoder (Encoder): The encoder model.
        decoder (Decoder): The decoder model.
        poly (Polynomial): The polynomial model.

    Methods:
        forward: Perform a forward pass through the model.
        renormalize_loss_weights: Renormalize the loss weights based on the current
            loss values so that they are accurately weighted based on the provided weights.
        total_loss: Calculate the total loss based on the loss weights.
        identity_loss: Calculate the identity loss (Encoder -> Decoder).
        l2_loss: Calculate the L2 loss.
        deriv_loss: Difference between the slopes of the predicted and true trajectories.
        deriv2_loss: Difference between the curvature of the predicted and true trajectories.
        deriv: Calculate the numerical derivative.
        deriv2: Calculate the numerical second derivative.
    """

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        if hasattr(config, "loss_weights"):
            self.loss_weights = config.loss_weights
        else:
            self.loss_weights = [100.0, 1.0, 1.0, 1.0]
        self.device = device

        self.encoder = Encoder(
            in_features=config.in_features,
            latent_features=config.latent_features,
            width_list=config.coder_layers,
            activation=config.activation,
        ).to(self.device)
        self.decoder = Decoder(
            out_features=config.in_features,
            latent_features=config.latent_features,
            width_list=config.coder_layers,
            activation=config.activation,
        ).to(self.device)
        self.poly = Polynomial(
            degree=self.config.degree, dimension=self.config.latent_features
        ).to(self.device)

    def forward(self, x, t_range):
        """
        Perform a forward pass through the model. Applies the encoder to the initial state,
        then propagates through time in the latent space by applying the polynomial to the time range.
        Finally, the decoder is applied to the latent space trajectory to obtain the predicted trajectory.

        Args:
            x (torch.Tensor): The input tensor.
            t_range (torch.Tensor): The time range to propagate through.

        Returns:
            torch.Tensor: The predicted trajectory.
        """
        current_batch_size = x.shape[0]
        x0 = x[:, 0, :]
        z0 = self.encoder(x0)  # x(t=0)
        t = t_range.unsqueeze(0).repeat(current_batch_size, 1)
        z_pred = self.poly(t) + z0.unsqueeze(1)
        return self.decoder(z_pred)

    def renormalize_loss_weights(self, x_true, x_pred):
        """
        Renormalize the loss weights based on the current loss values so that they are accurately
        weighted based on the provided weights. To be used once after a short burn in phase.

        Args:
            x_true (torch.Tensor): The true trajectory.
            x_pred (torch.Tensor): The predicted trajectory
        """
        self.loss_weights[0] = 1 / self.l2_loss(x_true, x_pred).item() * 100
        self.loss_weights[1] = 1 / self.identity_loss(x_true).item()
        self.loss_weights[2] = 1 / self.deriv_loss(x_true, x_pred).item()
        self.loss_weights[3] = 1 / self.deriv2_loss(x_true, x_pred).item()

    def total_loss(self, x_true, x_pred):
        """
        Calculate the total loss based on the loss weights.

        Args:
            x_true (torch.Tensor): The true trajectory.
            x_pred (torch.Tensor): The predicted trajectory

        Returns:
            torch.Tensor: The total loss.
        """
        return (
            self.loss_weights[0] * self.l2_loss(x_true, x_pred)
            + self.loss_weights[1] * self.identity_loss(x_true)
            + self.loss_weights[2] * self.deriv_loss(x_true, x_pred)
            + self.loss_weights[3] * self.deriv2_loss(x_true, x_pred)
        )

    def identity_loss(self, x: torch.Tensor):
        """
        Calculate the identity loss (Encoder -> Decoder).

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The identity loss.
        """
        return self.l2_loss(x, self.decoder(self.encoder(x)))

    @classmethod
    def l2_loss(cls, x_true: torch.Tensor, x_pred: torch.Tensor):
        """
        Calculate the L2 loss.

        Args:
            x_true (torch.Tensor): The true trajectory.
            x_pred (torch.Tensor): The predicted trajectory

        Returns:
            torch.Tensor: The L2 loss.
        """
        return torch.mean(torch.abs(x_true - x_pred) ** 2)

    @classmethod
    def deriv_loss(cls, x_true, x_pred):
        """
        Difference between the slopes of the predicted and true trajectories.

        Args:
            x_true (torch.Tensor): The true trajectory.
            x_pred (torch.Tensor): The predicted trajectory

        Returns:
            torch.Tensor: The derivative loss.
        """
        return cls.l2_loss(cls.deriv(x_pred), cls.deriv(x_true))

    @classmethod
    def deriv2_loss(cls, x_true, x_pred):
        """
        Difference between the curvature of the predicted and true trajectories.

        Args:
            x_true (torch.Tensor): The true trajectory.
            x_pred (torch.Tensor): The predicted trajectory

        Returns:
            torch.Tensor: The second derivative loss.
        """
        return cls.l2_loss(cls.deriv2(x_pred), cls.deriv2(x_true))

    @classmethod
    def deriv(cls, x):
        """
        Calculate the numerical derivative.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The numerical derivative.
        """
        return torch.gradient(x, dim=1)[0].squeeze(0)

    @classmethod
    def deriv2(cls, x):
        """
        Calculate the numerical second derivative.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The numerical second derivative.
        """
        return cls.deriv(cls.deriv(x))


class Polynomial(nn.Module):
    """
    Polynomial class with learnable parameters derived from nn.Module.

    Attributes:
        degree (int): the degree of the polynomial
        dimension (int): The dimension of the in- and output variables
        coef (nn.Linear): The linear layer for the polynomial coefficients
        t_matrix (torch.Tensor): The matrix of time values

    Methods:
        forward: Evaluate the polynomial at the given timesteps.
        _prepare_t: Prepare the time values in matrix form for the polynomial.
    """

    def __init__(self, degree: int, dimension: int):
        super().__init__()
        self.coef = nn.Linear(
            in_features=degree, out_features=dimension, bias=False, dtype=torch.float64
        )
        self.degree = degree
        self.dimension = dimension
        self.t_matrix = None

    def forward(self, t: torch.Tensor):
        """
        Evaluate the polynomial at the given timesteps.

        Args:
            t (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The evaluated polynomial.
        """
        # if self.t_matrix is None or self.t_matrix.shape[0] != t.shape[0]:
        #     self.t_matrix = self._prepare_t(t) # .clone()
        # return self.coef(self.t_matrix)
        return self.coef(self._prepare_t(t))

    def _prepare_t(self, t):
        """
        Prepare the time values in matrix form for the polynomial.

        Args:
            t (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The prepared time values.
        """
        t = t[:, None]
        return torch.hstack([t**i for i in range(1, self.degree + 1)]).permute(0, 2, 1)


AbstractSurrogateModel.register(LatentPoly)
