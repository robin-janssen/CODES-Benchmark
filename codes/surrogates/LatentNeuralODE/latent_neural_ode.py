from typing import Optional

import numpy as np
import torch
import torchode as to
from torch import Tensor, nn
from torch.utils.data import DataLoader

from codes.surrogates.AbstractSurrogate import AbstractSurrogateModel
from codes.utils import time_execution

from .latent_neural_ode_config import LatentNeuralODEBaseConfig
from .utilities import FlatSeqBatchIterable


class LatentNeuralODE(AbstractSurrogateModel):
    """
    LatentNeuralODE represents a latent neural ODE model.
    It includes an encoder, decoder, and neural ODE. Fixed parameters can be injected either
    into the encoder or later into the ODE network, controlled by config.encode_params.

    Args:
        device (str | None): Device for training (e.g. 'cpu', 'cuda:0').
        n_quantities (int): Number of quantities.
        n_timesteps (int): Number of timesteps.
        n_parameters (int): Number of fixed parameters (default 0).
        config (dict | None): Configuration for the model.
    """

    def __init__(
        self,
        device: str | None = None,
        n_quantities: int = 29,
        n_timesteps: int = 100,
        n_parameters: int = 0,
        training_id: str | None = None,
        config: dict | None = None,
    ):
        super().__init__(
            device=device,
            n_quantities=n_quantities,
            n_timesteps=n_timesteps,
            training_id=training_id,
            config=config,
        )
        self.config = LatentNeuralODEBaseConfig(**self.config)
        self.n_parameters = n_parameters
        # Instantiate the model wrapper with the additional n_parameters.
        self.model = ModelWrapper(
            config=self.config, n_quantities=n_quantities, n_parameters=n_parameters
        ).to(device)

    def forward(self, inputs):
        """
        Forward pass through the model.
        Expects inputs to be either (data, timesteps) or (data, timesteps, params).
        """
        inputs = tuple(
            (
                x.to(self.device, dtype=torch.float64, non_blocking=True)
                if isinstance(x, Tensor)
                else x
            )
            for x in inputs
        )
        x, t_range, params = inputs
        x0 = x[:, 0, :]
        latent_prediction = self.model(x0, t_range, params)
        return latent_prediction, x

    def prepare_data(
        self,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray | None,
        dataset_val: np.ndarray | None,
        timesteps: np.ndarray,
        batch_size: int = 128,
        shuffle: bool = True,
        dummy_timesteps: bool = True,
        dataset_train_params: np.ndarray | None = None,
        dataset_test_params: np.ndarray | None = None,
        dataset_val_params: np.ndarray | None = None,
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:

        if dummy_timesteps:
            timesteps = np.linspace(0, 1, dataset_train.shape[1])

        nw = getattr(self.config, "num_workers", 0)

        train_loader = self.create_dataloader(
            dataset_train,
            timesteps,
            batch_size,
            shuffle=shuffle,
            dataset_params=dataset_train_params,
            num_workers=nw,
        )

        test_loader = None
        if dataset_test is not None:
            test_loader = self.create_dataloader(
                dataset_test,
                timesteps,
                batch_size,
                shuffle=False,
                dataset_params=dataset_test_params,
                num_workers=nw,
            )

        val_loader = None
        if dataset_val is not None:
            val_loader = self.create_dataloader(
                dataset_val,
                timesteps,
                batch_size,
                shuffle=False,
                dataset_params=dataset_val_params,
                num_workers=nw,
            )

        return train_loader, test_loader, val_loader

    def create_dataloader(
        self,
        data: np.ndarray, 
        timesteps: np.ndarray, 
        batch_size: int,
        shuffle: bool,
        dataset_params: np.ndarray | None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        data_t = torch.from_numpy(data).float() 
        t_t = torch.from_numpy(timesteps).float()  
        if dataset_params is not None:
            params_t = torch.from_numpy(dataset_params).float()  
        else:
            params_t = None

        ds = FlatSeqBatchIterable(data_t, t_t, params_t, batch_size, shuffle)

        return DataLoader(
            ds,
            batch_size=None,
            num_workers=num_workers, 
            pin_memory=pin_memory,
            persistent_workers=False,
        )

    @time_execution
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        position: int = 0,
        description: str = "Training LatentNeuralODE",
        multi_objective: bool = False,
    ) -> None:
        """
        Train the LatentNeuralODE model.

        Args:
            train_loader (DataLoader): The data loader for the training data.
            test_loader (DataLoader): The data loader for the test data.
            epochs (int | None): The number of epochs to train the model. If None, uses the value from the config.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.
            multi_objective (bool): Whether multi-objective optimization is used.
                                    If True, trial.report is not used (not supported by Optuna).
        """
        optimizer, scheduler = self.setup_optimizer_and_scheduler(epochs)
        criterion = self.config.loss_function

        loss_length = (epochs + self.update_epochs - 1) // self.update_epochs
        self.train_loss, self.test_loss, self.MAE = [
            np.zeros(loss_length) for _ in range(3)
        ]

        progress_bar = self.setup_progress_bar(epochs, position, description)

        self.model.train()
        optimizer.train()

        self.setup_checkpoint()

        for epoch in progress_bar:
            for i, batch in enumerate(train_loader):
                batch = tuple(
                    (
                        x.to(device=self.device, non_blocking=True)
                        if isinstance(x, Tensor)
                        else x
                    )
                    for x in batch
                )
                x_true, t_range, params = batch
                optimizer.zero_grad()

                # forward pass
                x_pred, x_true = self((x_true, t_range, params))

                # total loss now takes params into account for identity term
                loss = self.model.total_loss(x_true, x_pred, params, criterion)
                loss.backward()
                optimizer.step()

                # renormalize once after 10 epochs
                if epoch == 10 and i == 0:
                    with torch.no_grad():
                        self.model.renormalize_loss_weights(
                            x_true, x_pred, params, criterion
                        )

            scheduler.step()

            self.validate(
                epoch=epoch,
                train_loader=train_loader,
                test_loader=test_loader,
                criterion=criterion,
                optimizer=optimizer,
                progress_bar=progress_bar,
                total_epochs=epochs,
                multi_objective=multi_objective,
            )

        progress_bar.close()
        self.n_epochs = epoch + 1
        self.get_checkpoint(test_loader, criterion)


class ModelWrapper(nn.Module):
    """
    Wraps the encoder, decoder, and neural ODE in three distinct modes:

    1. **No parameters** (n_parameters=0)
       - Encoder: input = state_dim
       - ODE: latent_dim -> latent_dim (the solver always evolves the latent state)
       - Decoder: latent_dim -> output dimensions

    2. **encode_params=True**
       - Encoder: input = state_dim + param_dim
       - ODE: latent_dim -> latent_dim
       - Decoder: latent_dim -> output dimensions

    3. **encode_params=False**
       - Encoder: input = state_dim
       - Base ODE: (latent_dim + param_dim) -> latent_dim
       - Wrapped in ODEWithParams so that solver state = latent_dim
       - Decoder: latent_dim -> output dimensions
    """

    def __init__(self, config, n_quantities: int, n_parameters: int = 0):
        super().__init__()
        self.config = config
        self.n_parameters = n_parameters
        latent_dim = config.latent_features
        self.loss_weights = getattr(config, "loss_weights", [100.0, 1.0, 1.0, 1.0])

        # --- Build encoder ---
        if n_parameters == 0:
            enc_in = n_quantities
        elif config.encode_params:
            enc_in = n_quantities + n_parameters
        else:
            enc_in = n_quantities
        self.encoder = Encoder(
            in_features=enc_in,
            latent_features=latent_dim,
            coder_layers=config.coder_layers,
            coder_width=config.coder_width,
            activation=config.activation,
        )

        # --- Build ODE ---
        if n_parameters == 0 or config.encode_params:
            # solver state = latent_dim
            ode_net = ODE(
                input_shape=latent_dim + (0 if config.encode_params else 0),
                output_shape=latent_dim,
                activation=config.activation,
                ode_layers=config.ode_layers,
                ode_width=config.ode_width,
                tanh_reg=config.ode_tanh_reg,
            )
            ode_module = ode_net
        else:
            # wrap base ODE to inject params: solver sees latent_dim state
            base_ode = ODE(
                input_shape=latent_dim + n_parameters,
                output_shape=latent_dim,
                activation=config.activation,
                ode_layers=config.ode_layers,
                ode_width=config.ode_width,
                tanh_reg=config.ode_tanh_reg,
            )
            ode_module = ODEWithParams(base_ode, n_parameters, latent_dim)

        self.ode = ode_module
        term = to.ODETerm(self.ode)
        step = to.Tsit5(term=term)
        ctrl = to.IntegralController(atol=config.atol, rtol=config.rtol, term=term)
        self.solver = to.AutoDiffAdjoint(step, ctrl)

        # --- Build decoder ---
        self.decoder = Decoder(
            out_features=n_quantities,
            latent_features=latent_dim,
            coder_layers=config.coder_layers,
            coder_width=config.coder_width,
            activation=config.activation,
        )

    def forward(self, x0: Tensor, t_range: Tensor, params: Tensor = None):
        # encode initial state
        if self.n_parameters > 0 and self.config.encode_params:
            assert params is not None
            enc_in = torch.cat([x0, params], dim=1)
        else:
            enc_in = x0
        z0 = self.encoder(enc_in)

        # if using closure to inject params
        if self.n_parameters > 0 and not self.config.encode_params:
            assert params is not None
            self.ode.set_params(params)

        # solve dynamics
        t_eval = t_range.repeat(x0.size(0), 1)
        sol = self.solver.solve(to.InitialValueProblem(y0=z0, t_eval=t_eval))
        latent_traj = sol.ys  # [timesteps, batch, latent_dim]

        # decode
        return self.decoder(latent_traj)

    def renormalize_loss_weights(
        self, x_true, x_pred, params, criterion: nn.Module = nn.MSELoss()
    ):
        """
        Renormalize the loss weights based on the current loss values so that they are accurately
        weighted based on the provided weights. To be used once after a short burn in phase.

        Args:
            x_true (Tensor): The true trajectory.
            x_pred (Tensor): The predicted trajectory
            params (Tensor): Fixed parameters (batch, n_parameters).
            criterion (nn.Module): Loss function to use for calculating the losses.
        """
        self.loss_weights[0] = 1 / criterion(x_pred, x_true).item() * 100
        self.loss_weights[1] = 1 / self.identity_loss(x_true, params).item()
        self.loss_weights[2] = 1 / self.deriv_loss(x_true, x_pred).item()
        self.loss_weights[3] = 1 / self.deriv2_loss(x_true, x_pred).item()

    def total_loss(
        self,
        x_true: Tensor,
        x_pred: Tensor,
        params: Tensor = None,
        criterion: nn.Module = nn.MSELoss(),
    ):
        """
        Calculate the total loss based on the loss weights, including params for identity.
        """
        return (
            self.loss_weights[0] * criterion(x_pred, x_true).item()
            + self.loss_weights[1] * self.identity_loss(x_true, params)
            + self.loss_weights[2] * self.deriv_loss(x_true, x_pred)
            + self.loss_weights[3] * self.deriv2_loss(x_true, x_pred)
        )

    def identity_loss(self, x_true: Tensor, params: Tensor = None):
        """
        Calculate the identity loss (Encoder -> Decoder) on the initial state x0.

        Args:
            x_true (Tensor): The full trajectory (batch, timesteps, features).
            params (Tensor | None): Fixed parameters (batch, n_parameters).
        Returns:
            Tensor: The identity loss on x0.
        """
        # only reconstruct the initial state
        x0 = x_true[:, 0, :]
        if self.config.encode_params and params is not None:
            enc_input = torch.cat([x0, params], dim=1)
        else:
            enc_input = x0

        # encode-decode
        z0 = self.encoder(enc_input)
        x0_hat = self.decoder(z0)
        return self.l2_loss(x0, x0_hat)

    @staticmethod
    def l2_loss(x_true: Tensor, x_pred: Tensor):
        """
        Calculate the L2 loss.

        Args:
            x_true (Tensor): The true trajectory.
            x_pred (Tensor): The predicted trajectory

        Returns:
            Tensor: The L2 loss.
        """
        return torch.mean(torch.abs(x_true - x_pred) ** 2)

    @classmethod
    def deriv_loss(cls, x_true, x_pred):
        """
        Difference between the slopes of the predicted and true trajectories.

        Args:
            x_true (Tensor): The true trajectory.
            x_pred (Tensor): The predicted trajectory

        Returns:
            Tensor: The derivative loss.
        """
        return cls.l2_loss(cls.deriv(x_pred), cls.deriv(x_true))

    @classmethod
    def deriv2_loss(cls, x_true, x_pred):
        """
        Difference between the curvature of the predicted and true trajectories.

        Args:
            x_true (Tensor): The true trajectory.
            x_pred (Tensor): The predicted trajectory

        Returns:
            Tensor: The second derivative loss.
        """
        return cls.l2_loss(cls.deriv2(x_pred), cls.deriv2(x_true))

    @staticmethod
    def deriv(x):
        """
        Calculate the numerical derivative.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The numerical derivative.
        """
        return torch.gradient(x, dim=1)[0].squeeze(0)

    @classmethod
    def deriv2(cls, x):
        """
        Calculate the numerical second derivative.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The numerical second derivative.
        """
        return cls.deriv(cls.deriv(x))


class ODE(nn.Module):
    """
    Neural ODE module defining the function for latent dynamics.
    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        activation: nn.Module,
        ode_layers: int,
        ode_width: int,
        tanh_reg: bool,
        dtype=torch.float64,
    ):
        super().__init__()
        self.tanh_reg = tanh_reg
        self.reg_factor = nn.Parameter(torch.tensor(1.0))
        self.activation = activation
        layers = []
        layers.append(nn.Linear(input_shape, ode_width, dtype=dtype))
        layers.append(activation)
        for _ in range(ode_layers):
            layers.append(nn.Linear(ode_width, ode_width, dtype=dtype))
            layers.append(activation)
        layers.append(nn.Linear(ode_width, output_shape, dtype=dtype))
        self.mlp = nn.Sequential(*layers)

    def forward(self, t, x):
        output = self.mlp(x)
        if self.tanh_reg:
            return self.reg_factor * torch.tanh(output / self.reg_factor)
        return output


class ODEWithParams(nn.Module):
    """
    Wraps a base ODE module so that parameters are injected as a constant.
    The solver sees only the latent state y (dim = latent_dim),
    but ODEWithParams.forward will concatenate y with p to compute dy/dt.
    """

    def __init__(self, base_ode: nn.Module, n_parameters: int, latent_dim: int):
        super().__init__()
        self.base_ode = base_ode
        self.latent_dim = latent_dim
        self.n_parameters = n_parameters
        self.p_const: Optional[Tensor] = None

    def set_params(self, params: Tensor):
        # params: [batch, n_parameters]
        self.p_const = params

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        # y: [batch, latent_dim]
        assert self.p_const is not None, "Call set_params() before solving."
        # concat state with params
        inp = torch.cat([y, self.p_const], dim=1)  # [batch, latent_dim + n_parameters]
        # compute derivative of latent only
        dy = self.base_ode(t, inp)  # [batch, latent_dim]
        return dy


class Encoder(nn.Module):
    """
    Fully connected encoder that maps input features to a latent space.
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int = 5,
        coder_layers: int = 3,
        coder_width: int = 32,
        activation: nn.Module = nn.ReLU(),
        dtype=torch.float64,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, coder_width, dtype=dtype))
        layers.append(activation)
        for _ in range(coder_layers - 1):
            layers.append(nn.Linear(coder_width, coder_width, dtype=dtype))
            layers.append(activation)
        layers.append(nn.Linear(coder_width, latent_features, dtype=dtype))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class Decoder(nn.Module):
    """
    Fully connected decoder that maps the latent space back to the output.
    """

    def __init__(
        self,
        out_features: int,
        latent_features: int = 5,
        coder_layers: int = 3,
        coder_width: int = 32,
        activation: nn.Module = nn.ReLU(),
        dtype=torch.float64,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(latent_features, coder_width, dtype=dtype))
        layers.append(activation)
        for _ in range(coder_layers - 1):
            layers.append(nn.Linear(coder_width, coder_width, dtype=dtype))
            layers.append(activation)
        layers.append(nn.Linear(coder_width, out_features, dtype=dtype))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class OldEncoder(nn.Module):
    """
    Old encoder using a fixed 4-2-1 structure.
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        # Example: coder_layers = [4, 2, 1] scaled by layers_factor.
        coder_layers = [4 * layers_factor, 2 * layers_factor, 1 * layers_factor]
        layers = []
        layers.append(nn.Linear(in_features, coder_layers[0], dtype=torch.float64))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[0], coder_layers[1], dtype=torch.float64))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[1], coder_layers[2], dtype=torch.float64))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[2], latent_features, dtype=torch.float64))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class OldDecoder(nn.Module):
    """
    Old decoder corresponding to the old encoder.
    """

    def __init__(
        self,
        out_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        coder_layers = [4 * layers_factor, 2 * layers_factor, 1 * layers_factor]
        # Reverse the order for the decoder.
        coder_layers.reverse()
        layers = []
        layers.append(nn.Linear(latent_features, coder_layers[0], dtype=torch.float64))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[0], coder_layers[1], dtype=torch.float64))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[1], coder_layers[2], dtype=torch.float64))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[2], out_features, dtype=torch.float64))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


AbstractSurrogateModel.register(LatentNeuralODE)
