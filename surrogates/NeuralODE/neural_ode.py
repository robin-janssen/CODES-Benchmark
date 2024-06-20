import os
import dataclasses
import yaml

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
from tqdm import tqdm

from surrogates.surrogates import AbstractSurrogateModel
from surrogates.NeuralODE.neural_ode_config import NeuralODEConfigOSU as Config
from utils import create_model_dir, time_execution
from surrogates.NeuralODE.utilities import ChemDataset


class NeuralODE(AbstractSurrogateModel):
    """
    NeuralODE is a class that represents a neural ordinary differential equation model.

    It inherits from the AbstractSurrogateModel class and implements methods for training,
    predicting, and saving the model.

    Attributes:
        model (ModelWrapper): The neural network model wrapped in a ModelWrapper object.
        train_loss (torch.Tensor): The training loss of the model.

    Methods:
        __init__(self, config: Config = Config()): Initializes a NeuralODE object.
        forward(self, inputs: torch.Tensor, timesteps: torch.Tensor):
            Performs a forward pass of the model.
        prepare_data(self, raw_data: np.ndarray, batch_size: int, shuffle: bool):
            Prepares the data for training and returns a Dataloader.
        fit(self, conf, data_loader, test_loader, timesteps, epochs): Trains the model.
        predict(self, data_loader): Makes predictions using the trained model.
        save(self, model_name: str, subfolder: str, training_id: str) -> None:
            Saves the model, losses, and hyperparameters.
    """

    def __init__(self, device: str | None = None):
        super().__init__()
        self.config: Config = Config()
        if device is not None:
            self.config.device = device
        self.device = self.config.device
        self.model = ModelWrapper(config=self.config).to(self.config.device)
        self.train_loss = None

    def forward(self, inputs: torch.Tensor, timesteps: torch.Tensor):
        """
        Perform a forward pass through the model.

        Args:
            inputs (torch.Tensor): The input tensor.
            timesteps (torch.Tensor): The tensor representing the timesteps.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.model.forward(inputs, timesteps)

    def prepare_data(
        self,
        timesteps: np.ndarray,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray | None = None,
        dataset_val: np.ndarray | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
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
        
        batch_size = self.config.batch_size if batch_size is None else batch_size
        device = self.config.device

        dset_train = ChemDataset(dataset_train, device=self.config.device)
        dataloader_train = torch.utils.data.DataLoader(
            dset_train, batch_size=batch_size, shuffle=shuffle
        )

        dataloader_test = None
        if dataset_test is not None:
            dset_test = ChemDataset(dataset_test, device=self.config.device)
            dataloader_test = torch.utils.data.DataLoader(
                dset_test, batch_size=batch_size, shuffle=shuffle
            )

        dataloader_val = None
        if dataset_val is not None:
            dset_val = ChemDataset(dataset_val, device=self.config.device)
            dataloader_val = torch.utils.data.DataLoader(
                dset_val, batch_size=batch_size, shuffle=shuffle
            )
        
        return dataloader_train, dataloader_test, dataloader_val
    


    @time_execution
    def fit(
        self,
        train_loader: DataLoader | Tensor,
        test_loader: DataLoader | Tensor,
        timesteps: np.ndarray,
        epochs: int | None,
    ):

        timesteps = torch.tensor(timesteps).to(self.config.device)
        epochs = self.config.epochs if epochs is None else epochs

        # TODO: make Optimizer and scheduler configable
        optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

        scheduler = None
        if self.config.final_learning_rate is not None:
            scheduler = CosineAnnealingLR(
                optimizer, self.config.epochs, eta_min=self.config.final_learning_rate
            )
        losses = torch.empty((self.config.epochs, len(train_loader)))
        # TODO: calculate test loss
        progress_bar = tqdm(range(epochs), desc="Training Progress")

        for epoch in progress_bar:
            for i, x_true in enumerate(train_loader):
                optimizer.zero_grad()
                x0 = x_true[:, 0, :]
                x_pred = self.model.forward(x0, timesteps)
                loss = self.model.total_loss(x_true, x_pred)
                loss.backward()
                optimizer.step()
                losses[epoch, i] = loss.item()

                if epoch == 10 and i == 0:
                    with torch.no_grad():
                        self.model.renormalize_loss_weights(x_true, x_pred)

            clr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix({"loss": losses[epoch, -1], "lr": clr})

            if scheduler is not None:
                scheduler.step()

            with torch.inference_mode():
                for i, x_true in enumerate(test_loader):

        self.train_loss = np.mean(losses, axis=1)

    def predict(
        self,
        data_loader: DataLoader | Tensor,
        criterion: nn.Module,
        timesteps: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:

        self.model.eval()
        self.model = self.model.to(self.config.device)

        total_loss = 0
        predictions = torch.empty((timesteps.shape[0], len(data_loader)))
        targets = torch.empty((timesteps.shape[0], len(data_loader)))
        if not isinstance(data_loader, DataLoader):
            raise TypeError("data_loader must be a DataLoader object")
        batch_size = data_loader.batch_size
        if batch_size is None:
            raise ValueError("batch_size must be provided by the DataLoader object")

        with torch.inference_mode():
            for i, x_true in enumerate(data_loader):
                x0 = x_true[:, 0, :]
                x_pred = self.model.forward(x0, timesteps)
                loss = criterion(x_true, x_pred)
                total_loss += loss.item()
                predictions[:, i * batch_size : (i + 1) * batch_size] = x_pred
                targets[:, i * batch_size : (i + 1) * batch_size] = x_true

        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        return total_loss, predictions, targets


class ModelWrapper(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_weights = [100.0, 1.0, 1.0, 1.0]

        self.encoder = Encoder(
            in_features=config.in_features,
            latent_features=config.latent_features,
            n_hidden=config.coder_hidden,
            width_list=config.coder_layers,
            activation=config.coder_activation,
        )
        self.decoder = Decoder(
            out_features=config.in_features,
            latent_features=config.latent_features,
            n_hidden=config.coder_hidden,
            width_list=config.coder_layers,
            activation=config.coder_activation,
        )
        self.ode = ODE(
            input_shape=config.latent_features,
            output_shape=config.latent_features,
            activation=config.ode_activation,
            n_hidden=config.ode_hidden,
            layer_width=config.ode_layer_width,
            tanh_reg=config.ode_tanh_reg,
        )

    def forward(self, x0, t_range):
        z0 = self.encoder(x0)  # x(t=0)
        if self.config.use_adjoint:
            result = odeint_adjoint(
                func=self.ode,
                y0=z0,
                t=t_range,
                adjoint_rtol=self.config.rtol,
                adjoint_atol=self.config.atol,
                adjoint_method=self.config.method,
            )
            if not isinstance(result, torch.Tensor):
                raise TypeError("odeint_adjoint must return tensor, check inputs")
            return self.decoder(torch.permute(result, dims=(1, 0, 2)))
        result = odeint(
            func=self.ode,
            y0=z0,
            t=t_range,
            rtol=self.config.rtol,
            atol=self.config.atol,
            method=self.config.method,
        )
        if not isinstance(result, torch.Tensor):
            raise TypeError("odeint must return tensor, check inputs")
        return self.decoder(torch.permute(result, dims=(1, 0, 2)))

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


class ODE(torch.nn.Module):

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        activation: torch.nn.Module,
        n_hidden: int,
        layer_width: int,
        tanh_reg: bool,
    ):
        super().__init__()

        self.tanh_reg = tanh_reg
        self.reg_factor = torch.nn.Parameter(torch.tensor(1.0))
        self.activation = activation

        self.mlp = torch.nn.Sequential()
        self.mlp.append(torch.nn.Linear(input_shape, layer_width, dtype=torch.float64))
        self.mlp.append(self.activation)
        for i in range(n_hidden):
            self.mlp.append(
                torch.nn.Linear(layer_width, layer_width, dtype=torch.float64)
            )
            self.mlp.append(self.activation)
        self.mlp.append(torch.nn.Linear(layer_width, output_shape, dtype=torch.float64))

    def forward(self, t, x):
        if self.tanh_reg:
            return self.reg_factor * torch.tanh(self.mlp(x) / self.reg_factor)
        return self.mlp(x)


class Encoder(torch.nn.Module):

    def __init__(
        self,
        in_features: int = 29,
        latent_features: int = 5,
        n_hidden: int = 4,
        width_list: list = [32, 16, 8],
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        assert (
            n_hidden == len(width_list) + 1
        ), "n_hidden must equal length of width_list"
        self.in_features = in_features
        self.latent_features = latent_features
        self.n_hidden = n_hidden
        self.width_list = width_list
        self.activation = activation

        self.mlp = torch.nn.Sequential()
        self.mlp.append(
            torch.nn.Linear(self.in_features, self.width_list[0], dtype=torch.float64)
        )
        self.mlp.append(self.activation)
        for i, width in enumerate(self.width_list[1:]):
            self.mlp.append(
                torch.nn.Linear(self.width_list[i], width, dtype=torch.float64)
            )
            self.mlp.append(self.activation)
        self.mlp.append(
            torch.nn.Linear(
                self.width_list[-1], self.latent_features, dtype=torch.float64
            )
        )
        self.mlp.append(torch.nn.Tanh())

    def forward(self, x):
        return self.mlp(x)


class Decoder(torch.nn.Module):

    def __init__(
        self,
        out_features: int,
        latent_features: int,
        n_hidden: int,
        width_list: list,
        activation: torch.nn.Module,
    ):
        super().__init__()
        assert (
            n_hidden == len(width_list) + 1
        ), "n_hidden must equal length of width_list"
        self.out_features = out_features
        self.latent_features = latent_features
        self.n_hidden = n_hidden
        self.width_list = width_list
        self.activation = activation
        self.width_list.reverse()

        self.mlp = torch.nn.Sequential()
        self.mlp.append(
            torch.nn.Linear(
                self.latent_features, self.width_list[0], dtype=torch.float64
            )
        )
        self.mlp.append(self.activation)
        for i, width in enumerate(self.width_list[1:]):
            self.mlp.append(
                torch.nn.Linear(self.width_list[i], width, dtype=torch.float64)
            )
            self.mlp.append(self.activation)
        self.mlp.append(
            torch.nn.Linear(self.width_list[-1], self.out_features, dtype=torch.float64)
        )
        self.mlp.append(torch.nn.Tanh())

    def forward(self, x):
        return self.mlp(x)