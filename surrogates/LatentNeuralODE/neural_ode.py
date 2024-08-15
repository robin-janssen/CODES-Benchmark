import numpy as np
import torch

# from torchdiffeq import odeint, odeint_adjoint
import torchode as to
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from surrogates.LatentNeuralODE.neural_ode_config import LatentNeuralODEBaseConfig
from surrogates.LatentNeuralODE.utilities import ChemDataset
from surrogates.surrogates import AbstractSurrogateModel
from utils import time_execution, worker_init_fn


class LatentNeuralODE(AbstractSurrogateModel):
    """
    LatentNeuralODE is a class that represents a neural ordinary differential equation model.

    It inherits from the AbstractSurrogateModel class and implements methods for training,
    predicting, and saving the model.

    Attributes:
        model (ModelWrapper): The neural network model wrapped in a ModelWrapper object.
        train_loss (torch.Tensor): The training loss of the model.

    Methods:
        __init__(self, config: Config = Config()): Initializes a LatentNeuralODE object.
        forward(self, inputs: torch.Tensor, timesteps: torch.Tensor):
            Performs a forward pass of the model.
        prepare_data(self, raw_data: np.ndarray, batch_size: int, shuffle: bool):
            Prepares the data for training and returns a Dataloader.
        fit(self, conf, data_loader, test_loader, timesteps, epochs): Trains the model.
        predict(self, data_loader): Makes predictions using the trained model.
        save(self, model_name: str, subfolder: str, training_id: str) -> None:
            Saves the model, losses, and hyperparameters.
    """

    def __init__(
        self,
        device: str | None = None,
        n_chemicals: int = 29,
        n_timesteps: int = 100,
        config: dict = {},
    ):
        super().__init__(
            device=device, n_chemicals=n_chemicals, n_timesteps=n_timesteps
        )
        self.config = LatentNeuralODEBaseConfig(**config)
        self.model = ModelWrapper(config=self.config, n_chemicals=n_chemicals).to(
            device
        )

    def forward(self, inputs):
        """
        Takes whatever the dataloader outputs, performs a forward pass through the
        model and returns the predictions with the respective targets.

        Args:
            inputs (Any): the data from the dataloader

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
        Prepares the data for training by creating a DataLoader object.

        Args:
            dataset (np.ndarray): The input dataset.
            timesteps (np.ndarray): The timesteps for the dataset.
            batch_size (int | None): The batch size for the DataLoader.
                If None, the entire dataset is loaded as a single batch.
            shuffle (bool): Whether to shuffle the data during training.

        Returns:
            DataLoader: The DataLoader object containing the prepared data.
        """
        device = self.device

        timesteps = torch.tensor(timesteps).to(device)

        dset_train = ChemDataset(dataset_train, timesteps, device=self.device)
        dataloader_train = DataLoader(
            dset_train,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
            collate_fn=lambda x: (x[0], x[1]),
        )

        dataloader_test = None
        if dataset_test is not None:
            dset_test = ChemDataset(dataset_test, timesteps, device=self.device)
            dataloader_test = DataLoader(
                dset_test,
                batch_size=batch_size,
                shuffle=shuffle,
                worker_init_fn=worker_init_fn,
                collate_fn=lambda x: (x[0], x[1]),
            )

        dataloader_val = None
        if dataset_val is not None:
            dset_val = ChemDataset(dataset_val, timesteps, device=device)
            dataloader_val = DataLoader(
                dset_val,
                batch_size=batch_size,
                shuffle=shuffle,
                worker_init_fn=worker_init_fn,
                collate_fn=lambda x: (x[0], x[1]),
            )

        return dataloader_train, dataloader_test, dataloader_val

    @time_execution
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        timesteps: np.ndarray | Tensor,
        epochs: int,
        position: int = 0,
        description: str = "Training LatentNeuralODE",
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
            timesteps = torch.tensor(timesteps).to(self.device)

        # TODO: make Optimizer and scheduler configable
        optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

        scheduler = None
        if self.config.final_learning_rate is not None:
            scheduler = CosineAnnealingLR(
                optimizer, epochs, eta_min=self.config.final_learning_rate
            )

        losses = torch.empty((epochs, len(train_loader)))
        test_losses = torch.empty((epochs))
        MAEs = torch.empty((epochs))

        progress_bar = self.setup_progress_bar(epochs, position, description)

        for epoch in progress_bar:
            for i, (x_true, timesteps) in enumerate(train_loader):
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
                preds, targets = self.predict(test_loader)
                self.model.train()
                loss = self.model.total_loss(preds, targets)
                test_losses[epoch] = loss
                MAEs[epoch] = self.L1(preds, targets).item()

        progress_bar.close()

        self.train_loss = torch.mean(losses, dim=1)
        self.test_loss = test_losses
        self.MAE = MAEs


class ModelWrapper(torch.nn.Module):

    def __init__(self, config, n_chemicals):
        super().__init__()
        self.config = config
        self.loss_weights = [100.0, 1.0, 1.0, 1.0]

        self.encoder = Encoder(
            in_features=n_chemicals,
            latent_features=config.latent_features,
            n_hidden=config.coder_hidden,
            width_list=config.coder_layers,
            activation=config.coder_activation,
        )
        self.decoder = Decoder(
            out_features=n_chemicals,
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
        term = to.ODETerm(self.ode)
        step_method = to.Tsit5(term=term)
        step_size_controller = to.IntegralController(
            atol=config.atol, rtol=config.rtol, term=term
        )
        self.solver = to.AutoDiffAdjoint(step_method, step_size_controller)

    def forward(self, x, t_range):
        x0 = x[:, 0, :]
        z0 = self.encoder(x0)  # x(t=0)
        t_eval = t_range.repeat(x.shape[0], 1)
        result = self.solver.solve(to.InitialValueProblem(y0=z0, t_eval=t_eval)).ys
        return self.decoder(result)

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
