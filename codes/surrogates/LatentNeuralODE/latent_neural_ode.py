from typing import Optional

import numpy as np
import torch
import torchode as to
from torch.utils.data import DataLoader

from codes.surrogates.AbstractSurrogate.surrogates import AbstractSurrogateModel
from codes.surrogates.LatentNeuralODE.latent_neural_ode_config import (
    LatentNeuralODEBaseConfig,
)
from codes.surrogates.LatentNeuralODE.utilities import ChemDataset
from codes.utils import time_execution, worker_init_fn


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
        # if len(inputs) == 3:
        x, t_range, params = inputs
        # else:
        #     x, t_range = inputs
        #     params = None

        # x has shape (batch, timesteps, n_quantities)
        # Use the first timestep as the initial condition.
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
        """
        Prepares data by creating DataLoader objects.
        If fixed parameters are provided, they are passed along with each sample.
        """
        if shuffle:
            shuffled_indices = np.random.permutation(len(dataset_train))
            dataset_train = dataset_train[shuffled_indices]
            if dataset_train_params is not None:
                dataset_train_params = dataset_train_params[shuffled_indices]

        if dummy_timesteps:
            timesteps = np.linspace(0, 1, dataset_train.shape[1])

        dset_train = ChemDataset(
            dataset_train,
            timesteps,
            device=self.device,
            parameters=dataset_train_params,
        )
        dataloader_train = DataLoader(
            dset_train,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            collate_fn=lambda x: (
                torch.stack([item[0] for item in x], dim=0),
                x[0][1],  # timesteps (assumed identical)
                torch.stack([item[2] for item in x], dim=0) if len(x[0]) == 3 else None,
            ),
        )

        dataloader_test = None
        if dataset_test is not None:
            dset_test = ChemDataset(
                dataset_test,
                timesteps,
                device=self.device,
                parameters=dataset_test_params,
            )
            dataloader_test = DataLoader(
                dset_test,
                batch_size=batch_size,
                shuffle=False,
                worker_init_fn=worker_init_fn,
                collate_fn=lambda x: (
                    torch.stack([item[0] for item in x], dim=0),
                    x[0][1],
                    (
                        torch.stack([item[2] for item in x], dim=0)
                        if len(x[0]) == 3
                        else None
                    ),
                ),
            )

        dataloader_val = None
        if dataset_val is not None:
            dset_val = ChemDataset(
                dataset_val,
                timesteps,
                device=self.device,
                parameters=dataset_val_params,
            )
            dataloader_val = DataLoader(
                dset_val,
                batch_size=batch_size,
                shuffle=False,
                worker_init_fn=worker_init_fn,
                collate_fn=lambda x: (
                    torch.stack([item[0] for item in x], dim=0),
                    x[0][1],
                    (
                        torch.stack([item[2] for item in x], dim=0)
                        if len(x[0]) == 3
                        else None
                    ),
                ),
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
        multi_objective: bool = False,
    ) -> None:
        """
        Fits the model to the training data. Sets the train_loss and test_loss attributes.
        After 10 epochs, the loss weights are renormalized to scale the individual loss terms.
        """
        optimizer, scheduler = self.setup_optimizer_and_scheduler(epochs)
        criterion = torch.nn.MSELoss()

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
                x_true, t_range, params = batch
                optimizer.zero_grad()

                # forward pass
                x_pred, _ = self.forward((x_true, t_range, params))

                # total loss now takes params into account for identity term
                loss = self.model.total_loss(x_true, x_pred, params)
                loss.backward()
                optimizer.step()

                # renormalize once after 10 epochs
                if epoch == 10 and i == 0:
                    with torch.no_grad():
                        self.model.renormalize_loss_weights(x_true, x_pred, params)

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


class ModelWrapper(torch.nn.Module):
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

    def forward(
        self, x0: torch.Tensor, t_range: torch.Tensor, params: torch.Tensor = None
    ):
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

    def renormalize_loss_weights(self, x_true, x_pred, params):
        """
        Renormalize the loss weights based on the current loss values so that they are accurately
        weighted based on the provided weights. To be used once after a short burn in phase.

        Args:
            x_true (torch.Tensor): The true trajectory.
            x_pred (torch.Tensor): The predicted trajectory
        """
        self.loss_weights[0] = 1 / self.l2_loss(x_true, x_pred).item() * 100
        self.loss_weights[1] = 1 / self.identity_loss(x_true, params).item()
        self.loss_weights[2] = 1 / self.deriv_loss(x_true, x_pred).item()
        self.loss_weights[3] = 1 / self.deriv2_loss(x_true, x_pred).item()

    def total_loss(
        self, x_true: torch.Tensor, x_pred: torch.Tensor, params: torch.Tensor = None
    ):
        """
        Calculate the total loss based on the loss weights, including params for identity.
        """
        return (
            self.loss_weights[0] * self.l2_loss(x_true, x_pred)
            + self.loss_weights[1] * self.identity_loss(x_true, params)
            + self.loss_weights[2] * self.deriv_loss(x_true, x_pred)
            + self.loss_weights[3] * self.deriv2_loss(x_true, x_pred)
        )

    def identity_loss(self, x_true: torch.Tensor, params: torch.Tensor = None):
        """
        Calculate the identity loss (Encoder -> Decoder) on the initial state x0.

        Args:
            x_true (torch.Tensor): The full trajectory (batch, timesteps, features).
            params (torch.Tensor | None): Fixed parameters (batch, n_parameters).
        Returns:
            torch.Tensor: The identity loss on x0.
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
    Neural ODE module defining the function for latent dynamics.
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
        output = self.mlp(x)
        if self.tanh_reg:
            return self.reg_factor * torch.tanh(output / self.reg_factor)
        return output


class ODEWithParams(torch.nn.Module):
    """
    Wraps a base ODE module so that parameters are injected as a constant.
    The solver sees only the latent state y (dim = latent_dim),
    but ODEWithParams.forward will concatenate y with p to compute dy/dt.
    """

    def __init__(self, base_ode: torch.nn.Module, n_parameters: int, latent_dim: int):
        super().__init__()
        self.base_ode = base_ode
        self.latent_dim = latent_dim
        self.n_parameters = n_parameters
        self.p_const: Optional[torch.Tensor] = None

    def set_params(self, params: torch.Tensor):
        # params: [batch, n_parameters]
        self.p_const = params

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y: [batch, latent_dim]
        assert self.p_const is not None, "Call set_params() before solving."
        # concat state with params
        inp = torch.cat([y, self.p_const], dim=1)  # [batch, latent_dim + n_parameters]
        # compute derivative of latent only
        dy = self.base_ode(t, inp)  # [batch, latent_dim]
        return dy


class Encoder(torch.nn.Module):
    """
    Fully connected encoder that maps input features to a latent space.
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
        return self.mlp(x)


class Decoder(torch.nn.Module):
    """
    Fully connected decoder that maps the latent space back to the output.
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
