import numpy as np
import optuna
import torch
from schedulefree import AdamWScheduleFree
from torch import nn

# from torch.optim import Adam
from torch.utils.data import DataLoader

from codes.surrogates.AbstractSurrogate.surrogates import AbstractSurrogateModel
from codes.surrogates.LatentNeuralODE.latent_neural_ode import Decoder as NewDecoder
from codes.surrogates.LatentNeuralODE.latent_neural_ode import Encoder as NewEncoder
from codes.surrogates.LatentNeuralODE.utilities import ChemDataset
from codes.surrogates.LatentPolynomial.latent_poly_config import (
    LatentPolynomialBaseConfig,
)
from codes.utils import time_execution, worker_init_fn


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
        config: dict | None = None,
    ):
        super().__init__(
            device=device,
            n_quantities=n_quantities,
            n_timesteps=n_timesteps,
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

    def forward(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the model.

        Args:
            inputs (tuple): Tuple containing the input tensor and timesteps.
                If fixed parameters are provided, the tuple is (data, timesteps, params).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (Predictions, Targets)
        """
        # Expect inputs to be (data, timesteps) or (data, timesteps, params)
        # if len(inputs) == 3:
        x, t_range, params = inputs
        # else:
        #     x, t_range = inputs
        #     params = None
        return self.model(x, t_range, params), x

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
        Prepare DataLoaders for training, testing, and validation.

        Args:
            dataset_train (np.ndarray): Training dataset.
            dataset_test (np.ndarray | None): Test dataset.
            dataset_val (np.ndarray | None): Validation dataset.
            timesteps (np.ndarray): Array of timesteps.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle training data.
            dummy_timesteps (bool): Whether to use dummy timesteps.
            dataset_*_params (np.ndarray | None): Fixed parameters for each split.

        Returns:
            tuple: DataLoaders for training, test, and validation datasets.
        """
        if dummy_timesteps:
            timesteps = np.linspace(0, 1, dataset_train.shape[1])
        if shuffle:
            shuffled_indices = np.random.permutation(len(dataset_train))
            dataset_train = dataset_train[shuffled_indices]
            if dataset_train_params is not None:
                dataset_train_params = dataset_train_params[shuffled_indices]
        # Note: ChemDataset already returns a 3‐tuple when parameters are provided.
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
                x[0][1],
                (
                    torch.stack([item[2] for item in x], dim=0)
                    if (len(x[0]) == 3 and x[0][2] is not None)
                    else None
                ),
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
                        if (len(x[0]) == 3 and x[0][2] is not None)
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
                        if (len(x[0]) == 3 and x[0][2] is not None)
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
        description: str = "Training LatentPoly",
        multi_objective: bool = False,
    ) -> None:
        """
        Fit the model to the training data.

        Args:
            train_loader (DataLoader): The data loader for the training data.
            test_loader (DataLoader): The data loader for the test data.
            epochs (int | None): The number of epochs to train the model. If None, uses the value from the config.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.
            multi_objective (bool): Whether multi-objective optimization is used.
                                    If True, trial.report is not used (not supported by Optuna).
        """
        optimizer = AdamWScheduleFree(
            self.model.parameters(), lr=self.config.learning_rate
        )

        loss_length = (epochs + self.update_epochs - 1) // self.update_epochs
        train_losses, test_losses, MAEs = [np.zeros(loss_length) for _ in range(3)]
        criterion = nn.MSELoss()

        progress_bar = self.setup_progress_bar(epochs, position, description)

        self.model.train()
        optimizer.train()

        for epoch in progress_bar:
            for i, batch in enumerate(train_loader):
                x_true, _, params = batch
                optimizer.zero_grad()
                x_pred, _ = self.forward(batch)
                loss = self.model.total_loss(x_true, x_pred, params)
                loss.backward()
                optimizer.step()

                if epoch == 10 and i == 0:
                    with torch.no_grad():
                        self.model.renormalize_loss_weights(x_true, x_pred, params)

            if epoch % self.update_epochs == 0:
                index = epoch // self.update_epochs
                with torch.inference_mode():
                    self.model.eval()
                    optimizer.eval()

                    preds, targets = self.predict(train_loader)
                    train_losses[index] = criterion(preds, targets).item()
                    preds, targets = self.predict(test_loader)
                    test_losses[index] = criterion(preds, targets).item()
                    MAEs[index] = self.L1(preds, targets).item()

                    progress_bar.set_postfix(
                        {
                            "train_loss": f"{train_losses[index]:.2e}",
                            "test_loss": f"{test_losses[index]:.2e}",
                        }
                    )

                    # Report loss to Optuna and prune if necessary
                    if self.optuna_trial is not None:
                        if multi_objective:
                            self.time_pruning(current_epoch=epoch, total_epochs=epochs)
                        else:
                            self.optuna_trial.report(test_losses[index], step=epoch)
                            if self.optuna_trial.should_prune():
                                raise optuna.TrialPruned()

                    self.model.train()
                    optimizer.train()

        progress_bar.close()

        self.n_epochs = epoch + 1
        self.train_loss = train_losses
        self.test_loss = test_losses
        self.MAE = MAEs


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
            ).to(self.device)
        # Instantiate the polynomial module.
        self.poly = Polynomial(degree=self.config.degree, dimension=latent_dim).to(
            self.device
        )
        # Instantiate coefficient network only if coeff_network is True and parameters exist.
        if self.config.coeff_network and n_parameters > 0:
            self.coefficient_net = nn.Sequential(
                nn.Linear(n_parameters, self.config.coeff_width, dtype=torch.float64),
                self.config.activation,
                *[
                    nn.Sequential(
                        nn.Linear(
                            self.config.coeff_width,
                            self.config.coeff_width,
                            dtype=torch.float64,
                        ),
                        self.config.activation,
                    )
                    for _ in range(self.config.coeff_layers - 1)
                ],
                nn.Linear(
                    self.config.coeff_width,
                    latent_dim * self.config.degree,
                    dtype=torch.float64,
                ),
            ).to(self.device)
        else:
            self.coefficient_net = None

    def forward(self, x, t_range, params=None):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, timesteps, n_quantities).
            t_range (torch.Tensor): Time range tensor.
            params (torch.Tensor | None): Fixed parameters of shape (batch, n_parameters), if provided.

        Returns:
            torch.Tensor: Predicted trajectory.
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

    def renormalize_loss_weights(self, x_true, x_pred, params=None):
        """
        Renormalize loss weights based on current loss values.

        Args:
            x_true (torch.Tensor): Ground truth.
            x_pred (torch.Tensor): Model predictions.
        """
        self.loss_weights[0] = 1 / self.l2_loss(x_true, x_pred).item() * 100
        self.loss_weights[1] = 1 / self.identity_loss(x_true, params).item()
        self.loss_weights[2] = 1 / self.deriv_loss(x_true, x_pred).item()
        self.loss_weights[3] = 1 / self.deriv2_loss(x_true, x_pred).item()

    def total_loss(
        self, x_true: torch.Tensor, x_pred: torch.Tensor, params: torch.Tensor = None
    ):
        """
        Compute the total loss, passing params into identity term.
        """
        return (
            self.loss_weights[0] * self.l2_loss(x_true, x_pred)
            + self.loss_weights[1] * self.identity_loss(x_true, params)
            + self.loss_weights[2] * self.deriv_loss(x_true, x_pred)
            + self.loss_weights[3] * self.deriv2_loss(x_true, x_pred)
        )

    def identity_loss(
        self, x_true: torch.Tensor, params: torch.Tensor = None
    ) -> torch.Tensor:
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
    def l2_loss(cls, x_true: torch.Tensor, x_pred: torch.Tensor):
        """
        Compute the L2 loss.

        Args:
            x_true (torch.Tensor): Ground truth.
            x_pred (torch.Tensor): Predictions.

        Returns:
            torch.Tensor: L2 loss.
        """
        return torch.mean(torch.abs(x_true - x_pred) ** 2)

    @classmethod
    def deriv_loss(cls, x_true, x_pred):
        """
        Compute the loss based on the difference of first derivatives.

        Args:
            x_true (torch.Tensor): Ground truth.
            x_pred (torch.Tensor): Predictions.

        Returns:
            torch.Tensor: Derivative loss.
        """
        return cls.l2_loss(cls.deriv(x_pred), cls.deriv(x_true))

    @classmethod
    def deriv2_loss(cls, x_true, x_pred):
        """
        Compute the loss based on the difference of second derivatives.

        Args:
            x_true (torch.Tensor): Ground truth.
            x_pred (torch.Tensor): Predictions.

        Returns:
            torch.Tensor: Second derivative loss.
        """
        return cls.l2_loss(cls.deriv2(x_pred), cls.deriv2(x_true))

    @classmethod
    def deriv(cls, x):
        """
        Compute the numerical first derivative.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: First derivative.
        """
        return torch.gradient(x, dim=1)[0].squeeze(0)

    @classmethod
    def deriv2(cls, x):
        """
        Compute the numerical second derivative.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Second derivative.
        """
        return cls.deriv(cls.deriv(x))


class Polynomial(nn.Module):
    """
    Learnable polynomial model.

    Attributes:
        degree (int): Degree of the polynomial.
        dimension (int): Dimension of the in- and output.
        coef (nn.Linear): Linear layer representing polynomial coefficients.
        t_matrix (torch.Tensor): Time matrix for polynomial evaluation.
    """

    def __init__(self, degree: int, dimension: int):
        super().__init__()
        self.degree = degree
        self.dimension = dimension
        self.coef = nn.Linear(
            in_features=degree, out_features=dimension, bias=False, dtype=torch.float64
        )
        self.t_matrix = None

    def forward(self, t: torch.Tensor):
        """
        Evaluate the polynomial at given timesteps.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Evaluated polynomial.
        """
        return self.coef(self._prepare_t(t))

    def _prepare_t(self, t: torch.Tensor):
        """
        Prepare time values in a matrix form.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Prepared time matrix.
        """
        t = t[:, None]
        return torch.hstack([t**i for i in range(1, self.degree + 1)]).permute(0, 2, 1)


class OldEncoder(torch.nn.Module):
    """
    Old encoder network implementing a fixed 4–2–1 structure scaled by layers_factor.

    Args:
        in_features (int): Number of input features.
        latent_features (int): Dimension of the latent space.
        layers_factor (int): Factor to scale the base widths [4, 2, 1].
        activation (torch.nn.Module): Activation function.
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        # Compute the hidden layer widths: [4, 2, 1] scaled by layers_factor.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the old encoder."""
        return self.mlp(x)


class OldDecoder(torch.nn.Module):
    """
    Old decoder network corresponding to the fixed 4–2–1 encoder.

    Args:
        out_features (int): Number of output features.
        latent_features (int): Dimension of the latent space.
        layers_factor (int): Factor to scale the base widths [4, 2, 1].
        activation (torch.nn.Module): Activation function.
    """

    def __init__(
        self,
        out_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        # Compute the hidden layer widths and reverse the order for decoding.
        coder_layers = [4 * layers_factor, 2 * layers_factor, 1 * layers_factor]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the old decoder."""
        return self.mlp(x)


AbstractSurrogateModel.register(LatentPoly)
