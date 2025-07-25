import numpy as np
import torch
from torch import Tensor, nn

# from torch.optim import Adam
from torch.utils.data import DataLoader

from codes.surrogates import Decoder as NewDecoder
from codes.surrogates import Encoder as NewEncoder
from codes.surrogates import FlatSeqBatchIterable
from codes.surrogates.AbstractSurrogate import AbstractSurrogateModel
from codes.utils import time_execution

from .latent_poly_config import LatentPolynomialBaseConfig


class LatentPoly(AbstractSurrogateModel):
    """
    LatentPoly class for training a polynomial model on latent space trajectories.

    This model includes an encoder, decoder, and a learnable polynomial applied on the latent space.
    The architecture is chosen based on the version flag in the configuration.

    Attributes:
        config (LatentPolynomialBaseConfig): The configuration for the model.
        model (PolynomialModelWrapper): The wrapped model (encoder, decoder, polynomial).
        device (str): Device for training.
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
        self.config = LatentPolynomialBaseConfig(**self.config)
        self.config.n_quantities = n_quantities
        # For backward compatibility: if using v1, compute the fixed width list.
        if self.config.model_version == "v1":
            coder_layers_old = [
                4 * self.config.layers_factor,
                2 * self.config.layers_factor,
                1 * self.config.layers_factor,
            ]
            self.config.width_list = coder_layers_old
        # # Update the number of input features based on whether we encode parameters.
        # if self.config.coeff_network:
        #     self.config.in_features = (
        #         n_quantities  # No parameter concatenation to the encoder input.
        #     )
        # else:
        #     self.config.in_features = n_quantities + n_parameters
        self.model = PolynomialModelWrapper(
            config=self.config, device=self.device, n_parameters=n_parameters
        )

    def forward(self, inputs) -> tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the model.

        Args:
            inputs (tuple): Tuple containing the input tensor and timesteps.
                If fixed parameters are provided, the tuple is (data, timesteps, params).

        Returns:
            tuple[Tensor, Tensor]: (Predictions, Targets)
        """
        inputs = tuple(
            x.to(self.device, non_blocking=True) if isinstance(x, Tensor) else x
            for x in inputs
        )
        x, t_range, params = inputs
        return self.model(x, t_range, params), x

    def create_dataloader(
        self,
        data: np.ndarray,  # (N, T, Q)
        timesteps: np.ndarray,  # (T,)
        batch_size: int,
        shuffle: bool,
        dataset_params: np.ndarray | None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        data_t = torch.from_numpy(data).float()
        t_t = torch.from_numpy(timesteps).float()
        params_t = (
            torch.from_numpy(dataset_params).float()
            if dataset_params is not None
            else None
        )

        ds = FlatSeqBatchIterable(data_t, t_t, params_t, batch_size, shuffle)

        return DataLoader(
            ds,
            batch_size=None,  # dataset yields full batches
            num_workers=num_workers,  # 0 usually fine
            pin_memory=pin_memory,
            persistent_workers=False,
        )

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
    ):
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
        Train the LatentPoly model.

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

        loss_length = (epochs + self.update_epochs - 1) // self.update_epochs
        self.train_loss, self.test_loss, self.MAE = [
            np.zeros(loss_length) for _ in range(3)
        ]
        criterion = nn.MSELoss()

        progress_bar = self.setup_progress_bar(epochs, position, description)

        self.model.train()
        optimizer.train()

        self.setup_checkpoint()

        for epoch in progress_bar:
            for i, batch in enumerate(train_loader):
                x_true, _, params = batch
                optimizer.zero_grad()
                x_pred, x_true = self(batch)
                loss = self.model.total_loss(x_true, x_pred, params)
                loss.backward()
                optimizer.step()

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
                optimizer=optimizer,
                progress_bar=progress_bar,
                total_epochs=epochs,
                multi_objective=multi_objective,
            )

        progress_bar.close()
        self.n_epochs = epoch + 1
        self.get_checkpoint(test_loader, criterion)


class PolynomialModelWrapper(nn.Module):
    """
    Wraps the Encoder, Decoder, and learnable Polynomial into a single model.

    If config.coeff_network is True, fixed parameters are not concatenated to the encoder input;
    instead, a coefficient network predicts the polynomial coefficients based on the parameters.
    If False, fixed parameters are concatenated to the encoder input and the polynomial coefficients are learned directly.

    Attributes:
        config (LatentPolynomialBaseConfig): Model configuration.
        loss_weights (list[float]): Weights for the loss terms.
        device (str): Device for training.
        encoder (Module): The encoder network.
        decoder (Module): The decoder network.
        poly (Polynomial): The polynomial module.
        coefficient_net (Module | None): The coefficient network (if config.coeff_network is True).
    """

    def __init__(self, config, device, n_parameters: int = 0):
        super().__init__()
        self.config = config
        self.loss_weights = getattr(config, "loss_weights", [100.0, 1.0, 1.0, 1.0])
        self.device = device
        latent_dim = self.config.latent_features

        # Use coeff_network as the single switch.
        if self.config.coeff_network and n_parameters > 0:
            encoder_in_features = self.config.n_quantities  # remains n_quantities
        else:
            # When not using the coefficient network, the encoder gets concatenated parameters.
            encoder_in_features = self.config.n_quantities + n_parameters

        # Instantiate encoder and decoder according to model_version.
        if self.config.model_version == "v1":
            self.encoder = OldEncoder(
                in_features=encoder_in_features,
                latent_features=latent_dim,
                layers_factor=self.config.layers_factor,
                activation=self.config.activation,
            ).to(self.device)
            # When parameters are concatenated, adjust the output features accordingly.
            # out_feats = (
            #     self.config.in_features
            #     if not self.config.coeff_network
            #     else self.config.in_features
            # )
            self.decoder = OldDecoder(
                out_features=self.config.n_quantities,
                latent_features=latent_dim,
                layers_factor=self.config.layers_factor,
                activation=self.config.activation,
            ).to(self.device)
        else:
            self.encoder = NewEncoder(
                in_features=encoder_in_features,
                latent_features=latent_dim,
                coder_layers=self.config.coder_layers,
                coder_width=self.config.coder_width,
                activation=self.config.activation,
                dtype=torch.float32,
            ).to(self.device)
            # out_feats = (
            #     self.config.in_features
            #     if not self.config.coeff_network
            #     else self.config.in_features
            # )
            self.decoder = NewDecoder(
                out_features=self.config.n_quantities,
                latent_features=latent_dim,
                coder_layers=self.config.coder_layers,
                coder_width=self.config.coder_width,
                activation=self.config.activation,
                dtype=torch.float32,
            ).to(self.device)
        # Instantiate the polynomial module.
        self.poly = Polynomial(degree=self.config.degree, dimension=latent_dim).to(
            self.device
        )
        # Instantiate coefficient network only if coeff_network is True and parameters exist.
        if self.config.coeff_network and n_parameters > 0:
            self.coefficient_net = nn.Sequential(
                nn.Linear(n_parameters, self.config.coeff_width, dtype=torch.float32),
                self.config.activation,
                *[
                    nn.Sequential(
                        nn.Linear(
                            self.config.coeff_width,
                            self.config.coeff_width,
                            dtype=torch.float32,
                        ),
                        self.config.activation,
                    )
                    for _ in range(self.config.coeff_layers - 1)
                ],
                nn.Linear(
                    self.config.coeff_width,
                    latent_dim * self.config.degree,
                    dtype=torch.float32,
                ),
            ).to(self.device)
        else:
            self.coefficient_net = None

    def forward(self, x, t_range, params=None):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (batch, timesteps, n_quantities).
            t_range (Tensor): Time range tensor.
            params (Tensor | None): Fixed parameters of shape (batch, n_parameters), if provided.

        Returns:
            Tensor: Predicted trajectory.
        """
        current_batch_size = x.shape[0]
        x0 = x[:, 0, :]
        if self.config.coeff_network and params is not None:
            # Scheme using coefficient network:
            encoder_input = x0  # Encoder gets only the raw data.
        else:
            # Otherwise, concatenate fixed parameters to the encoder input (if provided).
            if params is not None:
                encoder_input = torch.cat([x0, params], dim=1)
            else:
                encoder_input = x0
        z0 = self.encoder(encoder_input)
        t = t_range.unsqueeze(0).repeat(current_batch_size, 1)
        if self.config.coeff_network and (params is not None):
            # Use the coefficient network to predict polynomial coefficients.
            # Compute time basis using the internal _prepare_t method.
            B = self.poly._prepare_t(t)  # shape (batch, timesteps, degree)
            coef_vec = self.coefficient_net(params)  # shape (batch, latent_dim*degree)
            coef = coef_vec.view(current_batch_size, z0.shape[1], self.config.degree)
            # Multiply basis with predicted coefficients:
            poly_out = torch.bmm(
                B, coef.transpose(1, 2)
            )  # shape (batch, timesteps, latent_dim)
        else:
            poly_out = self.poly(t)
        z_pred = poly_out + z0.unsqueeze(1)
        return self.decoder(z_pred)

    def renormalize_loss_weights(
        self, x_true, x_pred, params, criterion: nn.Module = nn.MSELoss
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
            self.loss_weights[0] * criterion(x_pred, x_true)
            + self.loss_weights[1] * self.identity_loss(x_true, params)
            + self.loss_weights[2] * self.deriv_loss(x_true, x_pred)
            + self.loss_weights[3] * self.deriv2_loss(x_true, x_pred)
        )

    def identity_loss(self, x_true: Tensor, params: Tensor = None) -> Tensor:
        """
        Identity loss on the initial state x0, handling three cases:
          1. No params: params is None → encode x0 only.
          2. coeff_network=True: encode x0 only, ignore params here.
          3. coeff_network=False and params provided: encode [x0, params].
        """
        x0 = x_true[:, 0, :]  # [batch, n_quantities]
        # decide what to feed into the encoder:
        if params is None or self.config.coeff_network:
            enc_in = x0
        else:
            enc_in = torch.cat([x0, params], dim=1)
        z0 = self.encoder(enc_in)
        x0_hat = self.decoder(z0)
        return self.l2_loss(x0, x0_hat)

    @classmethod
    def l2_loss(cls, x_true: Tensor, x_pred: Tensor):
        """
        Compute the L2 loss.

        Args:
            x_true (Tensor): Ground truth.
            x_pred (Tensor): Predictions.

        Returns:
            Tensor: L2 loss.
        """
        return torch.mean(torch.abs(x_true - x_pred) ** 2)

    @classmethod
    def deriv_loss(cls, x_true, x_pred):
        """
        Compute the loss based on the difference of first derivatives.

        Args:
            x_true (Tensor): Ground truth.
            x_pred (Tensor): Predictions.

        Returns:
            Tensor: Derivative loss.
        """
        return cls.l2_loss(cls.deriv(x_pred), cls.deriv(x_true))

    @classmethod
    def deriv2_loss(cls, x_true, x_pred):
        """
        Compute the loss based on the difference of second derivatives.

        Args:
            x_true (Tensor): Ground truth.
            x_pred (Tensor): Predictions.

        Returns:
            Tensor: Second derivative loss.
        """
        return cls.l2_loss(cls.deriv2(x_pred), cls.deriv2(x_true))

    @classmethod
    def deriv(cls, x):
        """
        Compute the numerical first derivative.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: First derivative.
        """
        return torch.gradient(x, dim=1)[0].squeeze(0)

    @classmethod
    def deriv2(cls, x):
        """
        Compute the numerical second derivative.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Second derivative.
        """
        return cls.deriv(cls.deriv(x))


class Polynomial(nn.Module):
    """
    Learnable polynomial model.

    Attributes:
        degree (int): Degree of the polynomial.
        dimension (int): Dimension of the in- and output.
        coef (nn.Linear): Linear layer representing polynomial coefficients.
        t_matrix (Tensor): Time matrix for polynomial evaluation.
    """

    def __init__(self, degree: int, dimension: int):
        super().__init__()
        self.degree = degree
        self.dimension = dimension
        self.coef = nn.Linear(
            in_features=degree, out_features=dimension, bias=False, dtype=torch.float32
        )
        self.t_matrix = None

    def forward(self, t: Tensor):
        """
        Evaluate the polynomial at given timesteps.

        Args:
            t (Tensor): Time tensor.

        Returns:
            Tensor: Evaluated polynomial.
        """
        return self.coef(self._prepare_t(t))

    def _prepare_t(self, t: Tensor):
        """
        Prepare time values in a matrix form.

        Args:
            t (Tensor): Time tensor.

        Returns:
            Tensor: Prepared time matrix.
        """
        t = t[:, None]
        return torch.hstack([t**i for i in range(1, self.degree + 1)]).permute(0, 2, 1)


class OldEncoder(nn.Module):
    """
    Old encoder network implementing a fixed 4–2–1 structure scaled by layers_factor.

    Args:
        in_features (int): Number of input features.
        latent_features (int): Dimension of the latent space.
        layers_factor (int): Factor to scale the base widths [4, 2, 1].
        activation (nn.Module): Activation function.
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        # Compute the hidden layer widths: [4, 2, 1] scaled by layers_factor.
        coder_layers = [4 * layers_factor, 2 * layers_factor, 1 * layers_factor]
        layers = []
        layers.append(nn.Linear(in_features, coder_layers[0], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[0], coder_layers[1], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[1], coder_layers[2], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[2], latent_features, dtype=torch.float32))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the old encoder."""
        return self.mlp(x)


class OldDecoder(nn.Module):
    """
    Old decoder network corresponding to the fixed 4–2–1 encoder.

    Args:
        out_features (int): Number of output features.
        latent_features (int): Dimension of the latent space.
        layers_factor (int): Factor to scale the base widths [4, 2, 1].
        activation (nn.Module): Activation function.
    """

    def __init__(
        self,
        out_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        # Compute the hidden layer widths and reverse the order for decoding.
        coder_layers = [4 * layers_factor, 2 * layers_factor, 1 * layers_factor]
        coder_layers.reverse()
        layers = []
        layers.append(nn.Linear(latent_features, coder_layers[0], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[0], coder_layers[1], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[1], coder_layers[2], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[2], out_features, dtype=torch.float32))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the old decoder."""
        return self.mlp(x)


AbstractSurrogateModel.register(LatentPoly)
