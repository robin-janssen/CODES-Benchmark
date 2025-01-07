from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader, TensorDataset

from codes.surrogates.surrogates import AbstractSurrogateModel
from codes.utils import time_execution, worker_init_fn

from .deeponet_config import MultiONetBaseConfig
from .don_utils import mass_conservation_loss


class BranchNet(nn.Module):
    """
    Class that defines the branch network for the MultiONet model.

    Args:
        input_size (int): The input size for the network.
        hidden_size (int): The number of hidden units in each layer.
        output_size (int): The number of output units.
        num_hidden_layers (int): The number of hidden layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super(BranchNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), activation]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), activation]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the branch network.

        Args:
            x (torch.Tensor): The input tensor.
        """
        return self.network(x)


class TrunkNet(nn.Module):
    """
    Class that defines the trunk network for the MultiONet model.

    Args:
        input_size (int): The input size for the network.
        hidden_size (int): The number of hidden units in each layer.
        output_size (int): The number of output units.
        num_hidden_layers (int): The number of hidden layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super(TrunkNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), activation]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), activation]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the trunk network.

        Args:
            x (torch.Tensor): The input tensor.
        """
        return self.network(x)


class OperatorNetwork(AbstractSurrogateModel):
    """
    Abstract class for operator networks.
    Child classes must implement a forward method and use the branch_net and trunk_net attributes.

    Args:
        device (str, optional): The device to use for training (e.g., 'cpu', 'cuda:0').
        n_chemicals (int, optional): The number of chemicals.
        n_timesteps (int, optional): The number of timesteps.

    Raises:
        NotImplementedError: Child classes must implement a forward method.
        NotImplementedError: Child classes must initialize a branch_net and trunk_net.
    """

    def __init__(
        self,
        device: str | None = None,
        n_chemicals: int = 29,
        n_timesteps: int = 100,
        config: dict | None = None,
    ):
        super().__init__(
            device=device,
            n_chemicals=n_chemicals,
            n_timesteps=n_timesteps,
            config=config,
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
    """
    Class that implements the MultiONet model.
    It differs from a standard DeepONet in that it has multiple outputs, which are obtained by
    splitting the outputs of branch and trunk networks and calculating the scalar product of the splits.

    Args:
        device (str, optional): The device to use for training (e.g., 'cpu', 'cuda:0').
        n_chemicals (int, optional): The number of chemicals.
        n_timesteps (int, optional): The number of timesteps.
        config (dict, optional): The configuration for the model.

        The configuration must provide the following information:

        - trunk_input_size (int): The input size for the trunk network.
        - hidden_size (int): The number of hidden units in each layer of the branch and trunk networks.
        - branch_hidden_layers (int): The number of hidden layers in the branch network.
        - trunk_hidden_layers (int): The number of hidden layers in the trunk network.
        - output_factor (int): The factor by which the number of outputs is multiplied.
        - learning_rate (float): The learning rate for the optimizer.
        - schedule (bool): Whether to use a learning rate schedule.
        - regularization_factor (float): The regularization factor for the optimizer.
        - masses (np.ndarray, optional): The masses for mass conservation loss.
        - massloss_factor (float, optional): The factor for the mass conservation loss.

    Raises:
        TypeError: Invalid configuration for MultiONet model.
    """

    def __init__(
        self,
        device: str | None = None,
        n_chemicals: int = 29,
        n_timesteps: int = 100,
        config: dict | None = None,
    ):
        super().__init__(
            device=device,
            n_chemicals=n_chemicals,
            n_timesteps=n_timesteps,
            config=config,
        )
        self.config = MultiONetBaseConfig(**self.config)
        self.device = device
        self.N = n_chemicals  # Number of chemicals
        self.outputs = (
            n_chemicals * self.config.output_factor
        )  # Number of neurons in the last layer
        self.branch_net = BranchNet(
            n_chemicals - (self.config.trunk_input_size - 1),  # +1 due to time
            self.config.hidden_size,
            self.outputs,
            self.config.branch_hidden_layers,
            self.config.activation,
        ).to(device)
        self.trunk_net = TrunkNet(
            self.config.trunk_input_size,  # = time + optional additional quantities
            self.config.hidden_size,
            self.outputs,
            self.config.trunk_hidden_layers,
            self.config.activation,
        ).to(device)

    def forward(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MultiONet model.

        Args:
            inputs (tuple): The input tuple containing branch_input, trunk_input, and targets.

        Returns:
            tuple: The model outputs and the targets.
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
        epochs: int,
        position: int = 0,
        description: str = "Training DeepONet",
    ) -> None:
        """
        Train the MultiONet model.

        Args:
            train_loader (DataLoader): The DataLoader object containing the training data.
            test_loader (DataLoader): The DataLoader object containing the test data.
            epochs (int, optional): The number of epochs to train the model.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.

        Returns:
            None. The training loss, test loss, and MAE are stored in the model.
        """
        # self.n_timesteps = len(timesteps)
        self.n_train_samples = int(len(train_loader.dataset) / self.n_timesteps)

        criterion = self.setup_criterion()
        # optimizer = self.setup_optimizer_and_scheduler(epochs)
        optimizer = self.setup_optimizer_and_scheduler()

        train_losses, test_losses, MAEs = [np.zeros(epochs) for _ in range(3)]

        progress_bar = self.setup_progress_bar(epochs, position, description)

        for epoch in progress_bar:
            train_losses[epoch] = self.epoch(train_loader, criterion, optimizer)

            clr = optimizer.param_groups[0]["lr"]
            print_loss = f"{train_losses[epoch].item():.2e}"
            progress_bar.set_postfix({"loss": print_loss, "lr": f"{clr:.1e}"})
            # scheduler.step()

            if test_loader is not None:
                self.eval()
                optimizer.eval()
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
        # epochs: int,
        # ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    ) -> torch.optim.Optimizer:
        """
        Utility function to set up the optimizer and scheduler for training.

        Args:
            epochs (int): The number of epochs to train the model.

        Returns:
            tuple (torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler): The optimizer and scheduler.
        """
        # optimizer = torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.config.learning_rate,
        #     weight_decay=self.config.regularization_factor,
        # )
        # if self.config.schedule:
        #     scheduler = torch.optim.lr_scheduler.LinearLR(
        #         optimizer, start_factor=1, end_factor=0.3, total_iters=epochs
        #     )
        # else:
        #     scheduler = torch.optim.lr_scheduler.LinearLR(
        #         optimizer, start_factor=1, end_factor=1, total_iters=epochs
        #     )
        # return optimizer, scheduler
        optimizer = AdamWScheduleFree(
            self.parameters(),
            lr=self.config.learning_rate, 
        )
        return optimizer

    def epoch(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Perform one training epoch.

        Args:
            data_loader (DataLoader): The DataLoader object containing the training data.
            criterion (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            float: The total loss for the training step.
        """
        self.train()
        optimizer.train()
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
        data: np.ndarray,
        timesteps: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
    ):
        """
        Create a DataLoader for the given data.

        Args:
            data (np.ndarray): The data to load. Must have shape (n_samples, n_timesteps, n_chemicals).
            timesteps (np.ndarray): The timesteps.
            batch_size (int, optional): The batch size.
            shuffle (bool, optional): Whether to shuffle the data.
        """
        # Initialize lists to store the inputs and targets
        branch_inputs = []
        trunk_inputs = []
        targets = []

        # Iterate through the grid to select the samples
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                branch_inputs.append(data[i, 0, :])
                trunk_inputs.append([timesteps[j]])
                targets.append(data[i, j, :])

        # Convert to PyTorch tensors
        branch_inputs_tensor = torch.tensor(
            np.array(branch_inputs), dtype=torch.float32
        )
        trunk_inputs_tensor = torch.tensor(np.array(trunk_inputs), dtype=torch.float32)
        targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

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
        )
