from typing import Optional

import numpy as np
import optuna
import torch
import torchode as to
from schedulefree import AdamWScheduleFree
from torch.profiler import ProfilerActivity, profile, record_function

# from torch.optim import Adam
# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from codes.surrogates.AbstractSurrogate.surrogates import AbstractSurrogateModel
from codes.surrogates.LatentNeuralODE.latent_neural_ode_config import (
    LatentNeuralODEBaseConfig,
)
from codes.surrogates.LatentNeuralODE.utilities import ChemDataset
from codes.utils import time_execution, worker_init_fn


class LatentNeuralODE(AbstractSurrogateModel):
    """
    LatentNeuralODE is a class that represents a latent neural ordinary differential
    equation model. It includes an encoder, decoder, and neural ODE. The integrator is
    implemented by the torchode framework.

    Attributes:
        model (ModelWrapper): The neural network model wrapped in a ModelWrapper object.
        config (LatentNeuralODEBaseConfig): The configuration for the model.

    Methods:
        forward(inputs): Takes whatever the dataloader outputs, performs a forward pass
            through the model and returns the predictions with the respective targets.
        prepare_data(dataset_train, dataset_test, dataset_val, timesteps, batch_size,
            shuffle): Prepares the data for training by creating a DataLoader object.
        fit(train_loader, test_loader, epochs, position, description): Fits the model to
            the training data. Sets the train_loss and test_loss attributes.
    """

    def __init__(
        self,
        device: str | None = None,
        n_quantities: int = 29,
        n_timesteps: int = 100,
        model_config: dict | None = None,
    ):
        super().__init__(
            device=device,
            n_quantities=n_quantities,
            n_timesteps=n_timesteps,
            config=model_config,
        )
        self.config = LatentNeuralODEBaseConfig(**self.config)
        self.model = ModelWrapper(config=self.config, n_quantities=n_quantities).to(
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
        description: str = "Training LatentNeuralODE",
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
        """
        # optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        optimizer = AdamWScheduleFree(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        criterion = torch.nn.MSELoss()

        scheduler = None

        loss_length = (epochs + self.update_epochs - 1) // self.update_epochs
        train_losses, test_losses, MAEs = [np.zeros(loss_length) for _ in range(3)]

        progress_bar = self.setup_progress_bar(epochs, position, description)

        self.model.train()
        optimizer.train()

        for epoch in progress_bar:
            for i, (x_true, timesteps) in enumerate(train_loader):
                optimizer.zero_grad()
                x_pred = self.model.forward(x_true, timesteps)
                loss = self.model.total_loss(x_true, x_pred)
                loss.backward()
                optimizer.step()

                if epoch == 10 and i == 0:
                    with torch.no_grad():
                        self.model.renormalize_loss_weights(x_true, x_pred)

            if scheduler is not None:
                scheduler.step()

            if epoch % self.update_epochs == 0:
                index = epoch // self.update_epochs
                with torch.inference_mode():
                    # Set model and optimizer to evaluation mode
                    self.model.eval()
                    optimizer.eval()

                    # Calculate losses and MAE
                    preds, targets = self.predict(train_loader)
                    train_losses[index] = criterion(preds, targets).item()
                    preds, targets = self.predict(test_loader)
                    test_losses[index] = criterion(preds, targets).item()
                    MAEs[index] = self.L1(preds, targets).item()

                    # Update progress bar postfix
                    postfix = {
                        "train_loss": f"{train_losses[index]:.2e}",
                        "test_loss": f"{test_losses[index]:.2e}",
                    }
                    progress_bar.set_postfix(postfix)

                    # Report loss to Optuna and prune if necessary
                    if self.optuna_trial is not None:
                        self.optuna_trial.report(test_losses[index], step=epoch)
                        if self.optuna_trial.should_prune():
                            raise optuna.TrialPruned()

                    # Set model and optimizer back to training mode
                    self.model.train()
                    optimizer.train()

        progress_bar.close()

        self.n_epochs = epoch + 1
        self.train_loss = train_losses
        self.test_loss = test_losses
        self.MAE = MAEs

    @time_execution
    def fit_profile(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader],
        epochs: int,
        position: int = 0,
        description: str = "Training LatentNeuralODE with Profiling",
        profile_enabled: bool = True,  # Flag to enable/disable profiling
        profile_save_path: str = "chrome_trace_profile.json",  # Path to save Chrome trace
        profile_batches: int = 2,  # Number of batches to profile
        profile_epoch: int = 2,  # The epoch at which to perform profiling
    ) -> None:
        """
        Fits the model to the training data with optional profiling for a limited scope.
        Only used if renamed to fit in the main code (and renamed the original fit to something else).

        Args:
            train_loader (DataLoader): The data loader for the training data.
            test_loader (DataLoader | None): The data loader for the test data.
            epochs (int): The number of epochs to train the model.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.
            profile_enabled (bool): Whether to enable PyTorch profiling.
            profile_save_path (str): Path to save the profiling data.
            profile_batches (int): Number of batches to profile in the specified epoch.
            profile_epoch (int): The epoch at which profiling is performed.

        Returns:
            None. The training loss, test loss, and MAE are stored in the model.
        """
        optimizer = AdamWScheduleFree(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        optimizer.train()

        scheduler = None

        losses = torch.empty((epochs, len(train_loader)))
        test_losses = torch.empty((epochs))
        MAEs = torch.empty((epochs))

        progress_bar = self.setup_progress_bar(epochs, position, description)

        profiler = None
        if profile_enabled:
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )

        for epoch in progress_bar:
            with record_function("train_epoch"):
                if profile_enabled and epoch == profile_epoch:
                    profiler.start()
                    for i, (x_true, timesteps) in enumerate(train_loader):
                        if i >= profile_batches:
                            break
                        optimizer.zero_grad()
                        x_pred = self.model.forward(x_true, timesteps)
                        loss = self.model.total_loss(x_true, x_pred)
                        loss.backward()
                        optimizer.step()
                        losses[epoch, i] = loss.item()
                    profiler.stop()

                    # Print profiling summaries
                    print("\n### Profiling Summary ###\n")
                    print("\n### Key Averages (sorted by CUDA total time) ###\n")
                    print(
                        profiler.key_averages().table(
                            sort_by="cuda_time_total", row_limit=10
                        )
                    )
                    print("\n### Key Averages (sorted by CPU total time) ###\n")
                    print(
                        profiler.key_averages().table(
                            sort_by="cpu_time_total", row_limit=10
                        )
                    )
                    print("\n### Memory Usage Summary ###\n")
                    print(
                        profiler.key_averages().table(
                            sort_by="self_cuda_memory_usage", row_limit=10
                        )
                    )
                    profiler.export_chrome_trace(profile_save_path)
                    print(f"Chrome trace saved to '{profile_save_path}'")
                else:
                    # Normal training for all other epochs
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
            print_loss = (
                f"{losses[epoch, -1].item():.2e}" if len(train_loader) > 0 else "N/A"
            )
            progress_bar.set_postfix({"loss": print_loss, "lr": f"{clr:.1e}"})

            if scheduler is not None:
                scheduler.step()

            if test_loader is not None:
                with torch.inference_mode():
                    self.model.eval()
                    optimizer.eval()
                    preds, targets = self.predict(test_loader)
                    self.model.train()
                    optimizer.train()
                    loss = self.model.total_loss(preds, targets)
                    test_losses[epoch] = loss.item()
                    MAEs[epoch] = self.L1(preds, targets).item()

                    if self.optuna_trial is not None:
                        self.optuna_trial.report(loss.item(), epoch)
                        if self.optuna_trial.should_prune():
                            raise optuna.TrialPruned()

        if profiler is not None:
            profiler.shutdown()

        progress_bar.close()

        self.train_loss = torch.mean(losses, dim=1)
        self.test_loss = test_losses
        self.MAE = MAEs


class ModelWrapper(torch.nn.Module):
    """
    Wraps the encoder, decoder, and neural ODE into a single model.
    Chooses architecture based on the config.model_version flag.
    """
    def __init__(self, config, n_quantities: int):
        super().__init__()
        self.config = config
        self.loss_weights = [100.0, 1.0, 1.0, 1.0]

        # Conditional instantiation based on model_version.
        if config.model_version == "v1":
            # Instantiate the old encoder/decoder.
            self.encoder = OldEncoder(
                in_features=n_quantities,
                latent_features=config.latent_features,
                layers_factor=getattr(config, "layers_factor", 8),  # for backward compatibility
                activation=config.activation,
            )
            self.decoder = OldDecoder(
                out_features=n_quantities,
                latent_features=config.latent_features,
                layers_factor=getattr(config, "layers_factor", 8),
                activation=config.activation,
            )
        else:  # "v2" or any future version
            self.encoder = Encoder(
                in_features=n_quantities,
                latent_features=config.latent_features,
                coder_layers=config.coder_layers,
                coder_width=config.coder_width,
                activation=config.activation,
            )
            self.decoder = Decoder(
                out_features=n_quantities,
                latent_features=config.latent_features,
                coder_layers=config.coder_layers,
                coder_width=config.coder_width,
                activation=config.activation,
            )

        # The ODE part remains the same in both versions:
        self.ode = ODE(
            input_shape=config.latent_features,
            output_shape=config.latent_features,
            activation=config.activation,
            ode_layers=config.ode_layers,
            ode_width=config.ode_width,
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
        z0 = self.encoder(x0)
        t_eval = t_range.repeat(x.shape[0], 1)
        result = self.solver.solve(to.InitialValueProblem(y0=z0, t_eval=t_eval)).ys
        return self.decoder(result)

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

    @staticmethod
    def l2_loss(x_true: torch.Tensor, x_pred: torch.Tensor):
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

    @staticmethod
    def deriv(x):
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


class ODE(torch.nn.Module):
    """
    Neural ODE module that defines the ODE function for latent dynamics.

    The network is a feedforward network with a specified number of hidden layers (ode_layers)
    and uniform width (ode_width). Optionally applies a scaled tanh regularization.

    Args:
        input_shape (int): Input dimension (should match latent_features).
        output_shape (int): Output dimension (should match latent_features).
        activation (nn.Module): Activation function.
        ode_layers (int): Number of hidden layers.
        ode_width (int): Number of neurons in each hidden layer.
        tanh_reg (bool): Whether to apply scaled tanh regularization.
    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        activation: torch.nn.Module,
        ode_layers: int,
        ode_width: int,
        tanh_reg: bool,
    ):
        super().__init__()
        self.tanh_reg = tanh_reg
        self.reg_factor = torch.nn.Parameter(torch.tensor(1.0))
        self.activation = activation

        layers = []
        layers.append(torch.nn.Linear(input_shape, ode_width, dtype=torch.float64))
        layers.append(activation)
        for _ in range(ode_layers):
            layers.append(torch.nn.Linear(ode_width, ode_width, dtype=torch.float64))
            layers.append(activation)
        layers.append(torch.nn.Linear(ode_width, output_shape, dtype=torch.float64))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, t, x):
        """
        Forward pass for the ODE network.

        Args:
            t (torch.Tensor): Time tensor (unused in this implementation).
            x (torch.Tensor): Input latent state.

        Returns:
            torch.Tensor: Output latent state.
        """
        output = self.mlp(x)
        if self.tanh_reg:
            return self.reg_factor * torch.tanh(output / self.reg_factor)
        return output


class Encoder(torch.nn.Module):
    """
    Fully connected encoder network that maps input features to a lower-dimensional latent space.

    The architecture consists of a specified number of hidden layers (coder_layers) with uniform width (coder_width)
    and ends with a linear mapping to the latent space followed by a Tanh activation.

    Args:
        in_features (int): Number of input features.
        latent_features (int): Dimension of the latent representation.
        coder_layers (int): Number of hidden layers.
        coder_width (int): Number of neurons in each hidden layer.
        activation (nn.Module): Activation function.
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int = 5,
        coder_layers: int = 3,
        coder_width: int = 32,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        layers = []
        layers.append(torch.nn.Linear(in_features, coder_width, dtype=torch.float64))
        layers.append(activation)
        for _ in range(coder_layers - 1):
            layers.append(
                torch.nn.Linear(coder_width, coder_width, dtype=torch.float64)
            )
            layers.append(activation)
        layers.append(
            torch.nn.Linear(coder_width, latent_features, dtype=torch.float64)
        )
        layers.append(torch.nn.Tanh())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to encode the input into the latent space."""
        return self.mlp(x)


class Decoder(torch.nn.Module):
    """
    Fully connected decoder network that maps the latent representation back to the original output space.

    The network mirrors the encoder structure, using a specified number of hidden layers (coder_layers)
    with uniform width (coder_width) and ends with a linear mapping to the output features followed by Tanh.

    Args:
        out_features (int): Number of output features.
        latent_features (int): Dimension of the latent representation.
        coder_layers (int): Number of hidden layers.
        coder_width (int): Number of neurons in each hidden layer.
        activation (nn.Module): Activation function.
    """

    def __init__(
        self,
        out_features: int,
        latent_features: int = 5,
        coder_layers: int = 3,
        coder_width: int = 32,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        layers = []
        layers.append(
            torch.nn.Linear(latent_features, coder_width, dtype=torch.float64)
        )
        layers.append(activation)
        for _ in range(coder_layers - 1):
            layers.append(
                torch.nn.Linear(coder_width, coder_width, dtype=torch.float64)
            )
            layers.append(activation)
        layers.append(torch.nn.Linear(coder_width, out_features, dtype=torch.float64))
        layers.append(torch.nn.Tanh())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to decode the latent representation into output features."""
        return self.mlp(x)


class OldEncoder(torch.nn.Module):
    """
    Old encoder using a fixed 4-2-1 structure.
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        # Example: coder_layers = [4, 2, 1] scaled by layers_factor.
        coder_layers = [4 * layers_factor, 2 * layers_factor, 1 * layers_factor]
        layers = []
        layers.append(
            torch.nn.Linear(in_features, coder_layers[0], dtype=torch.float64)
        )
        layers.append(activation)
        layers.append(
            torch.nn.Linear(coder_layers[0], coder_layers[1], dtype=torch.float64)
        )
        layers.append(activation)
        layers.append(
            torch.nn.Linear(coder_layers[1], coder_layers[2], dtype=torch.float64)
        )
        layers.append(activation)
        layers.append(
            torch.nn.Linear(coder_layers[2], latent_features, dtype=torch.float64)
        )
        layers.append(torch.nn.Tanh())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class OldDecoder(torch.nn.Module):
    """
    Old decoder corresponding to the old encoder.
    """

    def __init__(
        self,
        out_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        coder_layers = [4 * layers_factor, 2 * layers_factor, 1 * layers_factor]
        # Reverse the order for the decoder.
        coder_layers.reverse()
        layers = []
        layers.append(
            torch.nn.Linear(latent_features, coder_layers[0], dtype=torch.float64)
        )
        layers.append(activation)
        layers.append(
            torch.nn.Linear(coder_layers[0], coder_layers[1], dtype=torch.float64)
        )
        layers.append(activation)
        layers.append(
            torch.nn.Linear(coder_layers[1], coder_layers[2], dtype=torch.float64)
        )
        layers.append(activation)
        layers.append(
            torch.nn.Linear(coder_layers[2], out_features, dtype=torch.float64)
        )
        layers.append(torch.nn.Tanh())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


AbstractSurrogateModel.register(LatentNeuralODE)
