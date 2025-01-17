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

from codes.surrogates.LatentNeuralODE.latent_neural_ode_config import (
    LatentNeuralODEBaseConfig,
)
from codes.surrogates.LatentNeuralODE.utilities import ChemDataset
from codes.surrogates.surrogates import AbstractSurrogateModel
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
        self.config = LatentNeuralODEBaseConfig(**self.config)
        coder_layers = [4, 2, 1]
        self.config.coder_layers = [
            layer * self.config.layers_factor for layer in coder_layers
        ]
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
            dataset_train (np.ndarray): The training dataset.
            dataset_test (np.ndarray): The test dataset.
            dataset_val (np.ndarray): The validation dataset.
            timesteps (np.ndarray): The array of timesteps.
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: The DataLoader object containing the prepared data.
        """

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
            dset_val = ChemDataset(dataset_val, timesteps, device=self.device)
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
        optimizer.train()

        scheduler = None
        # if self.config.final_learning_rate is not None:
        #     scheduler = CosineAnnealingLR(
        #         optimizer, epochs, eta_min=self.config.final_learning_rate
        #     )

        losses = torch.empty((epochs, len(train_loader)))
        test_losses = torch.empty((epochs))
        MAEs = torch.empty((epochs))

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
            print_loss = f"{losses[epoch, -1].item():.2e}"
            progress_bar.set_postfix({"loss": print_loss, "lr": f"{clr:.1e}"})

            if scheduler is not None:
                scheduler.step()

            with torch.inference_mode():
                self.model.eval()
                optimizer.eval()
                preds, targets = self.predict(test_loader)
                self.model.train()
                optimizer.train()
                loss = self.model.total_loss(preds, targets)
                test_losses[epoch] = loss
                MAEs[epoch] = self.L1(preds, targets).item()

                if self.optuna_trial is not None:
                    self.optuna_trial.report(loss, epoch)
                    if self.optuna_trial.should_prune():
                        raise optuna.TrialPruned()

        progress_bar.close()

        self.n_epochs = epoch
        self.train_loss = torch.mean(losses, dim=1)
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
        # Uncomment and configure the scheduler if needed
        # if self.config.final_learning_rate is not None:
        #     scheduler = CosineAnnealingLR(
        #         optimizer, epochs, eta_min=self.config.final_learning_rate
        #     )

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
    This class wraps the encoder, decoder and ODE term into a single model. It also
    provides the integration of the ODE term and the loss calculation.

    Attributes:
        config (LatentNeuralODEBaseConfig): The configuration for the model.
        loss_weights (list): The weights for the loss terms.
        encoder (Encoder): The encoder neural network.
        decoder (Decoder): The decoder neural network.
        ode (ODE): The neural ODE term.

    Methods:
        forward(x, t_range): Performs a forward pass through the model.
        renormalize_loss_weights(x_true, x_pred): Renormalizes the loss weights.
        total_loss(x_true, x_pred): Calculates the total loss.
        identity_loss(x): Calculates the identity loss (encoder -> decoder).
        l2_loss(x_true, x_pred): Calculates the L2 loss.
        deriv_loss(x_true, x_pred): Calculates the derivative loss.
        deriv2_loss(x_true, x_pred): Calculates the second derivative loss.
        deriv(x): Calculates the first derivative.
        deriv2(x): Calculates the second derivative.
    """

    def __init__(self, config, n_chemicals):
        super().__init__()
        self.config = config
        self.loss_weights = [100.0, 1.0, 1.0, 1.0]

        self.encoder = Encoder(
            in_features=n_chemicals,
            latent_features=config.latent_features,
            width_list=config.coder_layers,
            activation=config.activation,
        )
        self.decoder = Decoder(
            out_features=n_chemicals,
            latent_features=config.latent_features,
            width_list=config.coder_layers,
            activation=config.activation,
        )
        self.ode = ODE(
            input_shape=config.latent_features,
            output_shape=config.latent_features,
            n_hidden=config.ode_hidden,
            activation=config.activation,
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
        """
        Perform a forward pass through the model. Applies the encoder to the initial state,
        then propagates through time in the latent space by integrating the neural ODE term.
        Finally, the decoder is applied to the latent state to obtain the predicted trajectory.

        Args:
            x (torch.Tensor): The input tensor.
            t_range (torch.Tensor): The range of timesteps.

        Returns:
            torch.Tensor: The predicted trajectory.
        """
        x0 = x[:, 0, :]
        z0 = self.encoder(x0)  # x(t=0)
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
    The neural ODE term. The term itself is a simple feedforward neural network,
    a scaled tanh function is applied to the output if tanh_reg is set to True.

    Attributes:
        tanh_reg (bool): Whether to apply a tanh regularization to the output.
        reg_factor (torch.Tensor): The regularization factor.
        activation (torch.nn.Module): The activation function.
        mlp (torch.nn.Sequential): The neural network.

    Methods:
        forward(t, x): Perform a forward pass through the neural network.
    """

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
        for _ in range(n_hidden):
            self.mlp.append(
                torch.nn.Linear(layer_width, layer_width, dtype=torch.float64)
            )
            self.mlp.append(self.activation)
        self.mlp.append(torch.nn.Linear(layer_width, output_shape, dtype=torch.float64))

    def forward(self, t, x):
        """
        The forward pass through the neural network.

        Args:
            t (torch.Tensor): The time tensor.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the neural network.
        """
        if self.tanh_reg:
            return self.reg_factor * torch.tanh(self.mlp(x) / self.reg_factor)
        return self.mlp(x)


class Encoder(torch.nn.Module):
    """
    The encoder neural network. The encoder is a simple feedforward neural network
    the output of which is of a lower dimension than the input.

    Attributes:
        in_features (int): The number of input features.
        latent_features (int): The number of latent features.
        n_hidden (int): The number of hidden layers.
        width_list (list): The width of the hidden layers.
        activation (torch.nn.Module): The activation function.
        mlp (torch.nn.Sequential): The neural network.

    Methods:
        forward(x): Perform a forward pass through the neural network. ("Encode" the input)
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int = 5,
        width_list: list | None = None,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        self.width_list = width_list if width_list is not None else [32, 16, 8]
        self.n_hidden = len(self.width_list) + 1
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
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the neural network. ("Encoded" input)
        """
        return self.mlp(x)


class Decoder(torch.nn.Module):
    """
    The decoder neural network. The decoder is a simple feedforward neural network
    the output of which is of a higher dimension than the input. Acts as the approximate
    inverse of the encoder.

    Attributes:
        out_features (int): The number of output features.
        latent_features (int): The number of latent features.
        width_list (list): The width of the hidden layers.
        activation (torch.nn.Module): The activation function.
        mlp (torch.nn.Sequential): The neural network.

    Methods:
        forward(x): Perform a forward pass through the neural network. ("Decode" the input)
    """

    def __init__(
        self,
        out_features: int,
        latent_features: int = 5,
        width_list: list | None = None,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        self.out_features = out_features
        self.latent_features = latent_features
        self.width_list = width_list if width_list is not None else [32, 16, 8]
        self.n_hidden = len(self.width_list) + 1
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
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the neural network. ("Decoded" input)
        """
        return self.mlp(x)


AbstractSurrogateModel.register(LatentNeuralODE)
