import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import Tensor
from torch.optim import Adam
import numpy as np

from surrogates.surrogates import AbstractSurrogateModel
from surrogates.LatentPolynomial.latent_poly_config import LatentPolynomialConfigOSU
from surrogates.NeuralODE.utilities import ChemDataset
from surrogates.NeuralODE.neural_ode import Encoder, Decoder
from utils import time_execution


class LatentPoly(AbstractSurrogateModel):

    def __init__(self, device: str | None = None, N_chemicals: int = 29):
        super().__init__(device=device, N_chemicals=N_chemicals)
        self.config: LatentPolynomialConfigOSU = LatentPolynomialConfigOSU()
        self.config.in_features = N_chemicals
        self.model = PolynomialModelWrapper(config=self.config, device=self.device)

    def forward(self, inputs: Tensor, timesteps: Tensor | np.ndarray):
        """
        Perform a forward pass through the model.

        Args:
            inputs (torch.Tensor): The input tensor.
            timesteps (torch.Tensor | np.ndarray): The tensor/array representing the timesteps.

        Returns:
            torch.Tensor: The output tensor.
        """
        if not isinstance(timesteps, Tensor):
            timesteps = torch.tensor(timesteps, dtype=torch.float64).to(self.device)
        return self.model.forward(inputs, timesteps)

    def prepare_data(
        self,
        timesteps: np.ndarray,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray | None = None,
        dataset_val: np.ndarray | None = None,
        batch_size: int = 128,
        shuffle: bool = True,
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        """
        Prepares the data for training by creating a DataLoader object.

        Args:
            dataset (np.ndarray): The input dataset.
            timesteps (np.ndarray): The timesteps for the dataset.
            batch_size (int | None): The batch size for the DataLoader. If None, the entire dataset is loaded as a single batch.
            shuffle (bool): Whether to shuffle the data during training.

        Returns:
            DataLoader: The DataLoader object containing the prepared data.
        """
        device = self.device

        dset_train = ChemDataset(dataset_train, device=self.device)
        dataloader_train = DataLoader(
            dset_train, batch_size=batch_size, shuffle=shuffle
        )

        dataloader_test = None
        if dataset_test is not None:
            dset_test = ChemDataset(dataset_test, device=self.device)
            dataloader_test = DataLoader(
                dset_test, batch_size=batch_size, shuffle=shuffle
            )

        dataloader_val = None
        if dataset_val is not None:
            dset_val = ChemDataset(dataset_val, device=device)
            dataloader_val = DataLoader(
                dset_val, batch_size=batch_size, shuffle=shuffle
            )

        return dataloader_train, dataloader_test, dataloader_val

    @time_execution
    def fit(
        self,
        train_loader: DataLoader | Tensor,
        test_loader: DataLoader | Tensor,
        timesteps: np.ndarray | Tensor,
        epochs: int,
        position: int = 0,
        description: str = "Training LatentPoly",
    ) -> None:
        """
        Fits the model to the training data. Sets the train_loss and test_loss attributes.

        Args:
            train_loader (DataLoader): The data loader for the training data.
            test_loader (DataLoader): The data loader for the test data.
            timesteps (np.ndarray | Tensor): The array of timesteps.
            epochs (int | None): The number of epochs to train the model. If None, uses the value from the config.

        Returns:
            None
        """
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps).to(self.device)

        # TODO: make Optimizer and scheduler configable
        optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

        losses = torch.empty((epochs, len(train_loader)))
        test_losses = torch.empty((epochs))
        accuracy = torch.empty((epochs))

        progress_bar = self.setup_progress_bar(epochs, position, description)

        for epoch in progress_bar:

            for i, x_true in enumerate(train_loader):
                optimizer.zero_grad()
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

            with torch.inference_mode():
                self.model.eval()
                preds, targets = self.predict(test_loader, timesteps)
                self.model.train()
                loss = self.model.total_loss(preds, targets)
                test_losses[epoch] = loss
                accuracy[epoch] = 1.0 - torch.mean(
                    torch.abs(preds - targets) / torch.abs(targets)
                )

        progress_bar.close()

        self.train_loss = torch.mean(losses, dim=1)
        self.test_loss = test_losses
        self.accuracy = accuracy

    def predict(
        self,
        data_loader: DataLoader,
        timesteps: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Makes predictions using the trained model.

        Args:
            data_loader (DataLoader): The DataLoader object containing the data.
            timesteps (np.ndarray | torch.Tensor): The array of timesteps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the predictions and the targets.
        """
        self.model = self.model.to(self.device)
        if not isinstance(timesteps, torch.Tensor):
            t_range = torch.tensor(timesteps).to(self.device)
        else:
            t_range = timesteps

        batch_size = data_loader.batch_size
        if batch_size is None:
            raise ValueError("batch_size must be provided by the DataLoader object")

        predictions = torch.empty_like(data_loader.dataset.data)  # type: ignore
        targets = torch.empty_like(data_loader.dataset.data)  # type: ignore

        with torch.inference_mode():
            for i, x_true in enumerate(data_loader):
                x_pred = self.model.forward(x_true, t_range)
                predictions[i * batch_size : (i + 1) * batch_size, :, :] = x_pred
                targets[i * batch_size : (i + 1) * batch_size, :, :] = x_true

        preds = self.denormalize(predictions)
        targets = self.denormalize(targets)

        return preds, targets


class PolynomialModelWrapper(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.loss_weights = [100.0, 1.0, 1.0, 1.0]
        self.device = device

        self.encoder = Encoder(
            in_features=config.in_features,
            latent_features=config.latent_features,
            n_hidden=config.coder_hidden,
            width_list=config.coder_layers,
            activation=config.coder_activation,
        ).to(self.device)
        self.decoder = Decoder(
            out_features=config.in_features,
            latent_features=config.latent_features,
            n_hidden=config.coder_hidden,
            width_list=config.coder_layers,
            activation=config.coder_activation,
        ).to(self.device)
        self.poly = Polynomial(
            degree=self.config.degree, dimension=self.config.latent_features
        ).to(self.device)

    def forward(self, x, t_range):
        current_batch_size = x.shape[0]
        x0 = x[:, 0, :]
        z0 = self.encoder(x0)  # x(t=0)
        t = t_range.unsqueeze(0).repeat(current_batch_size, 1)
        z_pred = self.poly(t) + z0.unsqueeze(1)
        return self.decoder(z_pred)

    def renormalize_loss_weights(self, x_true, x_pred):
        self.loss_weights[0] = 1 / self.l2_loss(x_true, x_pred).item() * 100
        self.loss_weights[1] = 1 / self.identity_loss(x_true).item()
        self.loss_weights[2] = 1 / self.deriv_loss(x_true, x_pred).item()
        self.loss_weights[3] = 1 / self.deriv2_loss(x_true, x_pred).item()

    def total_loss(self, x_true, x_pred):
        return (
            self.loss_weights[0] * self.l2_loss(x_true, x_pred)
            + self.loss_weights[1] * self.identity_loss(x_true)
            + self.loss_weights[2] * self.deriv_loss(x_true, x_pred)
            + self.loss_weights[3] * self.deriv2_loss(x_true, x_pred)
        )

    def identity_loss(self, x: torch.Tensor):
        return self.l2_loss(x, self.decoder(self.encoder(x)))

    @classmethod
    def l2_loss(cls, x_true: torch.Tensor, x_pred: torch.Tensor):
        return torch.mean(torch.abs(x_true - x_pred) ** 2)

    @classmethod
    def mass_conservation_loss(cls):
        raise NotImplementedError("Don't use yet please")

    @classmethod
    def deriv_loss(cls, x_true, x_pred):
        return cls.l2_loss(cls.deriv(x_pred), cls.deriv(x_true))

    @classmethod
    def deriv2_loss(cls, x_true, x_pred):
        return cls.l2_loss(cls.deriv2(x_pred), cls.deriv2(x_true))

    @classmethod
    def deriv(cls, x):
        return torch.gradient(x, dim=1)[0].squeeze(0)

    @classmethod
    def deriv2(cls, x):
        return cls.deriv(cls.deriv(x))


class Polynomial(nn.Module):
    """
    Polynomial class with learnable parameters

    Attributes:
        degree (int): the degree of the polynomial
        dimension (int): The dimension of the in- and output variables
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
        if self.t_matrix is None or self.t_matrix.shape[0] != t.shape[0]:
            self.t_matrix = self._prepare_t(t)
        return self.coef(self.t_matrix)

    def _prepare_t(self, t):
        t = t[:, None]
        return torch.hstack([t**i for i in range(1, self.degree + 1)]).permute(0, 2, 1)
