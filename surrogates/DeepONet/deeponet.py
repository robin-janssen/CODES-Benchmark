from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Use the below import to adjust the config class to the specific model
from surrogates.DeepONet.deeponet_config import OChemicalTrainConfig as MultiONetConfig
from surrogates.surrogates import AbstractSurrogateModel
from utils import time_execution, worker_init_fn

from .utils import mass_conservation_loss


class BranchNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(BranchNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TrunkNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(TrunkNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class OperatorNetwork(AbstractSurrogateModel):
    def __init__(
        self, device: str | None = None, n_chemicals: int = 29, n_timesteps: int = 100
    ):
        super().__init__(
            device=device, n_chemicals=n_chemicals, n_timesteps=n_timesteps
        )

    def post_init_check(self):
        if not hasattr(self, "branch_net") or not hasattr(self, "trunk_net"):
            raise NotImplementedError(
                "Child classes must initialize a branch_net and trunk_net."
            )
        if not hasattr(self, "forward") or not callable(self.forward):
            raise NotImplementedError("Child classes must implement a forward method.")

    def forward(self, branch_input, trunk_input):
        # Define a generic forward pass or raise an error to enforce child class implementation
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    @staticmethod
    def _calculate_split_sizes(total_neurons, num_splits):
        """Helper function to calculate split sizes for even distribution"""
        base_size = total_neurons // num_splits
        remainder = total_neurons % num_splits
        return [
            base_size + 1 if i < remainder else base_size for i in range(num_splits)
        ]


OperatorNetworkType = TypeVar("OperatorNetworkType", bound=OperatorNetwork)


class MultiONet(OperatorNetwork):
    def __init__(
        self, device: str | None = None, n_chemicals: int = 29, n_timesteps: int = 100
    ):
        """
        Initialize the MultiONet model with a configuration.

        The configuration must provide the following information:

        - trunk_input_size (int): The input size for the trunk network.
        - hidden_size (int): The number of hidden units in each layer of the branch and trunk networks.
        - branch_hidden_layers (int): The number of hidden layers in the branch network.
        - trunk_hidden_layers (int): The number of hidden layers in the trunk network.
        - output_neurons (int): The number of neurons in the last layer of both branch and trunk networks.
        - N_outputs (int): The number of outputs of the model.
        - device (str): The device to use for training (e.g., 'cpu', 'cuda:0').
        """
        config = MultiONetConfig()  # Load the specific config for DeepONet
        # super(MultiONet, self).__init__()
        super().__init__(
            device=device, n_chemicals=n_chemicals, n_timesteps=n_timesteps
        )

        self.config = config
        self.device = device
        self.N = n_chemicals  # Number of chemicals
        self.outputs = (
            n_chemicals * config.output_factor
        )  # Number of neurons in the last layer
        self.branch_net = BranchNet(
            n_chemicals - config.trunk_input_size + 1,  # +1 due to time
            config.hidden_size,
            self.outputs,
            config.branch_hidden_layers,
        ).to(device)
        self.trunk_net = TrunkNet(
            config.trunk_input_size,  # = time + optional additional quantities
            config.hidden_size,
            self.outputs,
            config.trunk_hidden_layers,
        ).to(device)

    def forward(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MultiONet model.

        Args:
            inputs (tuple): The input tuple containing branch_input, trunk_input, and targets.
            Note: The targets are not used in the forward pass, but are included for compatibility with DataLoader.
            timesteps (np.ndarray, optional): The timesteps.
            Note: The timesteps are not used in the forward pass, but are included for compatibility with the benchmarking code.

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        branch_input, trunk_input, targets = inputs
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        # Splitting the outputs for multiple output values
        split_sizes = self._calculate_split_sizes(self.outputs, self.N)
        branch_splits = torch.split(branch_output, split_sizes, dim=1)
        trunk_splits = torch.split(trunk_output, split_sizes, dim=1)

        result = []
        for b_split, t_split in zip(branch_splits, trunk_splits):
            result.append(torch.sum(b_split * t_split, dim=1, keepdim=True))

        return torch.cat(result, dim=1), targets

    def prepare_data(
        self,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray,
        dataset_val: np.ndarray | None,
        timesteps: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader | None]:
        """
        Prepare the data for the predict or fit methods.
        Note: All datasets must have shape (n_samples, n_timesteps, n_chemicals).

        Args:
            dataset_train (np.ndarray): The training data.
            dataset_test (np.ndarray): The test data.
            dataset_val (np.ndarray, optional): The validation data.
            timesteps (np.ndarray): The timesteps.
            batch_size (int, optional): The batch size.
            shuffle (bool, optional): Whether to shuffle the data.

        Returns:
            tuple: The training, test, and validation DataLoaders.
        """
        dataloaders = []
        # Create the train dataloader
        dataloader_train = self.create_dataloader(
            dataset_train,
            timesteps,
            batch_size,
            shuffle,
            train=True,
        )
        dataloaders.append(dataloader_train)

        # Create the test and validation dataloaders
        for dataset in [dataset_test, dataset_val]:
            if dataset is not None:
                dataloader = self.create_dataloader(
                    dataset,
                    timesteps,
                    batch_size,
                    shuffle,
                )
                dataloaders.append(dataloader)
            else:
                dataloaders.append(None)

        return dataloaders[0], dataloaders[1], dataloaders[2]

    @time_execution
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        timesteps: np.ndarray,
        epochs: int,
        position: int = 0,
        description: str = "Training DeepONet",
    ) -> None:
        """
        Train the MultiONet model.

        Args:
            train_data (np.ndarray): The training data.
            test_data (np.ndarray): The test data (to evaluate the model during training).
            timesteps (np.ndarray): The timesteps.
            epochs (int, optional): The number of epochs to train the model.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.

        Returns:
            None
        """
        self.n_timesteps = len(timesteps)
        self.n_train_samples = int(len(train_loader.dataset) / self.n_timesteps)

        criterion = self.setup_criterion()
        optimizer, scheduler = self.setup_optimizer_and_scheduler(epochs)

        train_losses, test_losses, MAEs = [np.zeros(epochs) for _ in range(3)]

        progress_bar = self.setup_progress_bar(epochs, position, description)

        for epoch in progress_bar:
            train_losses[epoch] = self.epoch(train_loader, criterion, optimizer)

            clr = optimizer.param_groups[0]["lr"]
            print_loss = f"{train_losses[epoch].item():.2e}"
            progress_bar.set_postfix({"loss": print_loss, "lr": f"{clr:.1e}"})
            scheduler.step()

            if test_loader is not None:
                preds, targets = self.predict(test_loader)
                test_losses[epoch] = criterion(preds, targets).item() / torch.numel(
                    targets
                )
                MAEs[epoch] = self.L1(preds, targets).item()

        progress_bar.close()

        self.train_loss = train_losses
        self.test_loss = test_losses
        self.MAE = MAEs

    def setup_criterion(self) -> callable:
        """
        Utility function to set up the loss function for training.

        Returns:
            callable: The loss function.
        """
        crit = nn.MSELoss(reduction="sum")
        if hasattr(self.config, "masses") and self.config.masses is not None:
            weights = (1.0, self.config.massloss_factor)
            crit = mass_conservation_loss(
                self.config.masses, crit, weights, self.device
            )
        return crit

    def setup_optimizer_and_scheduler(
        self,
        epochs: int,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        Utility function to set up the optimizer and scheduler for training.

        Args:
            epochs (int): The number of epochs to train the model.

        Returns:
            tuple (torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler): The optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.regularization_factor,
        )
        if self.config.schedule:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1, end_factor=0.3, total_iters=epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1, end_factor=1, total_iters=epochs
            )
        return optimizer, scheduler

    def epoch(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Perform a single training step on the model.

        Args:
            data_loader (DataLoader): The DataLoader object containing the training data.
            criterion (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            float: The total loss for the training step.
        """
        self.train()
        total_loss = 0
        dataset_size = len(data_loader.dataset)

        for batch in data_loader:
            branch_input, trunk_input, targets = batch
            branch_input, trunk_input, targets = (
                branch_input.to(self.device),
                trunk_input.to(self.device),
                targets.to(self.device),
            )
            optimizer.zero_grad()
            outputs, targets = self((branch_input, trunk_input, targets))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= dataset_size * self.N
        return total_loss

    def create_dataloader(
        self,
        data,
        timesteps,
        batch_size=32,
        shuffle: bool = False,
        normalize: bool = True,
        fraction: float = 1,
        train: bool = False,
    ):
        """
        Create a DataLoader with optional fractional subsampling for chemical evolution data for DeepONet.

        :param data: 3D numpy array with shape (num_samples, len(timesteps), num_chemicals)
        :param timesteps: 1D numpy array of timesteps.
        :param fraction: Fraction of the grid points to sample.
        :param batch_size: Batch size for the DataLoader.
        :param shuffle: Whether to shuffle the data.
        :param normalize: Whether to normalize the data.#
        :param device: Device to use.
        :param train: Whether the provided data is training data. If so, the mean and standard deviation are computed.
        :return: A DataLoader object.
        """
        # Initialize lists to store the inputs and targets
        branch_inputs = []
        trunk_inputs = []
        targets = []

        # Iterate through the grid to select the samples
        if fraction == 1:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    branch_inputs.append(data[i, 0, :])
                    trunk_inputs.append([timesteps[j]])
                    targets.append(data[i, j, :])
        else:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if np.random.uniform(0, 1) < fraction:
                        branch_inputs.append(data[i, :, 0])
                        trunk_inputs.append([timesteps[j]])
                        targets.append(data[i, :, j])

        # Convert to PyTorch tensors
        branch_inputs_tensor = torch.tensor(
            np.array(branch_inputs), dtype=torch.float32
        )
        trunk_inputs_tensor = torch.tensor(np.array(trunk_inputs), dtype=torch.float32)
        targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

        if train:
            self.b_mean = branch_inputs_tensor.mean()
            self.b_std = branch_inputs_tensor.std()
            self.target_mean = targets_tensor.mean()
            self.target_std = targets_tensor.std()

        if normalize:
            branch_inputs_tensor = (branch_inputs_tensor - self.b_mean) / self.b_std
            targets_tensor = (targets_tensor - self.target_mean) / self.target_std

        # Create a TensorDataset and DataLoader
        branch_inputs_tensor = branch_inputs_tensor.to(self.device)
        trunk_inputs_tensor = trunk_inputs_tensor.to(self.device)
        targets_tensor = targets_tensor.to(self.device)
        dataset = TensorDataset(
            branch_inputs_tensor, trunk_inputs_tensor, targets_tensor
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
            # num_workers=4,
        )
