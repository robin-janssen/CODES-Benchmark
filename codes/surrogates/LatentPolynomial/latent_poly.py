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
        model_config: dict | None = None,
    ):
        super().__init__(
            device=device,
            n_quantities=n_quantities,
            n_timesteps=n_timesteps,
            config=model_config,
        )
        self.config = LatentPolynomialBaseConfig(**self.config)
        # For backward compatibility: if using v1, compute the fixed width list.
        if self.config.model_version == "v1":
            # Compute width list for v1 using the fixed [4, 2, 1] structure scaled by layers_factor.
            coder_layers_old = [
                self.config.coder_hidden,
                self.config.coder_hidden // 2,
                self.config.coder_hidden // 4,
            ]
            # Alternatively, if your old logic multiplies fixed numbers by layers_factor:
            coder_layers_old = [
                4 * self.config.layers_factor,
                2 * self.config.layers_factor,
                1 * self.config.layers_factor,
            ]
            self.config.width_list = coder_layers_old
        # Set the number of input features.
        self.config.in_features = n_quantities
        self.model = PolynomialModelWrapper(config=self.config, device=self.device)

    def forward(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the model.

        Args:
            inputs (tuple): Tuple containing the input tensor and timesteps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (Predictions, Targets)
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
        Prepare DataLoaders for training, testing, and validation.

        Args:
            dataset_train (np.ndarray): Training dataset.
            dataset_test (np.ndarray | None): Test dataset.
            dataset_val (np.ndarray | None): Validation dataset.
            timesteps (np.ndarray): Array of timesteps.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle training data.

        Returns:
            tuple: DataLoaders for training, test, and validation datasets.
        """
        if shuffle:
            shuffled_indices = np.random.permutation(len(dataset_train))
            dataset_train = dataset_train[shuffled_indices]

        dset_train = ChemDataset(dataset_train, timesteps, device=self.device)
        dataloader_train = DataLoader(
            dset_train,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            collate_fn=lambda x: (x[0], x[1]),
        )

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
    ) -> None:
        """
        Fit the model to the training data.

        Args:
            train_loader (DataLoader): Training data loader.
            test_loader (DataLoader): Test data loader.
            epochs (int): Number of training epochs.
            position (int): Progress bar position.
            description (str): Description for the progress bar.
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
            for i, (x_true, timesteps) in enumerate(train_loader):
                optimizer.zero_grad()
                x_pred = self.model.forward(x_true, timesteps)
                loss = self.model.total_loss(x_true, x_pred)
                loss.backward()
                optimizer.step()

                if epoch == 10 and i == 0:
                    with torch.no_grad():
                        self.model.renormalize_loss_weights(x_true, x_pred)

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

                    if self.optuna_trial is not None:
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

    The correct encoder/decoder architecture is chosen based on the model_version flag
    in the configuration (v1 uses the old fixed structure; v2 uses the new FCNN design).

    Attributes:
        config (LatentPolynomialBaseConfig): Model configuration.
        loss_weights (list[float]): Weights for the loss terms.
        device (str): Device for training.
        encoder (Module): The encoder network.
        decoder (Module): The decoder network.
        poly (Polynomial): The polynomial model.
    """

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.loss_weights = getattr(config, "loss_weights", [100.0, 1.0, 1.0, 1.0])
        self.device = device

        # Conditional instantiation based on model_version.
        if config.model_version == "v1":
            self.encoder = OldEncoder(
                in_features=config.in_features,
                latent_features=config.latent_features,
                layers_factor=config.layers_factor,
                activation=config.activation,
            ).to(self.device)
            self.decoder = OldDecoder(
                out_features=config.in_features,
                latent_features=config.latent_features,
                layers_factor=config.layers_factor,
                activation=config.activation,
            ).to(self.device)
        else:
            self.encoder = NewEncoder(
                in_features=config.in_features,
                latent_features=config.latent_features,
                coder_layers=config.coder_layers,
                coder_width=config.coder_width,
                activation=config.activation,
            ).to(self.device)
            self.decoder = NewDecoder(
                out_features=config.in_features,
                latent_features=config.latent_features,
                coder_layers=config.coder_layers,
                coder_width=config.coder_width,
                activation=config.activation,
            ).to(self.device)
        self.poly = Polynomial(
            degree=self.config.degree, dimension=self.config.latent_features
        ).to(self.device)

    def forward(self, x, t_range):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.
            t_range (torch.Tensor): Time range tensor.

        Returns:
            torch.Tensor: Predicted trajectory.
        """
        current_batch_size = x.shape[0]
        x0 = x[:, 0, :]
        z0 = self.encoder(x0)
        t = t_range.unsqueeze(0).repeat(current_batch_size, 1)
        # Apply the polynomial and add the initial latent state.
        z_pred = self.poly(t) + z0.unsqueeze(1)
        return self.decoder(z_pred)

    def renormalize_loss_weights(self, x_true, x_pred):
        """
        Renormalize loss weights based on current loss values.

        Args:
            x_true (torch.Tensor): Ground truth.
            x_pred (torch.Tensor): Model predictions.
        """
        self.loss_weights[0] = 1 / self.l2_loss(x_true, x_pred).item() * 100
        self.loss_weights[1] = 1 / self.identity_loss(x_true).item()
        self.loss_weights[2] = 1 / self.deriv_loss(x_true, x_pred).item()
        self.loss_weights[3] = 1 / self.deriv2_loss(x_true, x_pred).item()

    def total_loss(self, x_true, x_pred):
        """
        Compute the total loss as a weighted sum of loss terms.

        Args:
            x_true (torch.Tensor): Ground truth.
            x_pred (torch.Tensor): Predictions.

        Returns:
            torch.Tensor: Total loss.
        """
        return (
            self.loss_weights[0] * self.l2_loss(x_true, x_pred)
            + self.loss_weights[1] * self.identity_loss(x_true)
            + self.loss_weights[2] * self.deriv_loss(x_true, x_pred)
            + self.loss_weights[3] * self.deriv2_loss(x_true, x_pred)
        )

    def identity_loss(self, x: torch.Tensor):
        """
        Compute the identity loss between inputs and decoded outputs.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Identity loss.
        """
        return self.l2_loss(x, self.decoder(self.encoder(x)))

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
