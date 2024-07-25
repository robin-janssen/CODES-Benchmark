import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, TypeVar

from surrogates.surrogates import AbstractSurrogateModel

# Use the below import to adjust the config class to the specific model
from surrogates.DeepONet.deeponet_config import OChemicalTrainConfig as MultiONetConfig

from utils import time_execution
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
    def __init__(self):
        super(OperatorNetwork, self).__init__()

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
    def __init__(self, device: str | None = None):
        """
        Initialize the MultiONet model with a configuration.

        The configuration must provide the following information:

        - branch_input_size (int): The input size for the branch network.
        - trunk_input_size (int): The input size for the trunk network.
        - hidden_size (int): The number of hidden units in each layer of the branch and trunk networks.
        - branch_hidden_layers (int): The number of hidden layers in the branch network.
        - trunk_hidden_layers (int): The number of hidden layers in the trunk network.
        - output_neurons (int): The number of neurons in the last layer of both branch and trunk networks.
        - N_outputs (int): The number of outputs of the model.
        - device (str): The device to use for training (e.g., 'cpu', 'cuda:0').
        """
        config = MultiONetConfig()  # Load the specific config for DeepONet
        super(MultiONet, self).__init__()

        self.config = config
        if device is not None:
            config.device = device
        self.device = config.device
        self.N = config.N_outputs  # Number of outputs
        self.outputs = config.output_neurons  # Number of neurons in the last layer
        self.branch_net = BranchNet(
            config.branch_input_size,
            config.hidden_size,
            config.output_neurons,
            config.branch_hidden_layers,
        ).to(config.device)
        self.trunk_net = TrunkNet(
            config.trunk_input_size,
            config.hidden_size,
            config.output_neurons,
            config.trunk_hidden_layers,
        ).to(config.device)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        timesteps: np.ndarray | None = None,
    ) -> torch.Tensor:
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
        branch_input, trunk_input, _ = inputs
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        # Splitting the outputs for multiple output values
        split_sizes = self._calculate_split_sizes(self.outputs, self.N)
        branch_splits = torch.split(branch_output, split_sizes, dim=1)
        trunk_splits = torch.split(trunk_output, split_sizes, dim=1)

        result = []
        for b_split, t_split in zip(branch_splits, trunk_splits):
            result.append(torch.sum(b_split * t_split, dim=1, keepdim=True))

        return torch.cat(result, dim=1)

    def prepare_data(
        self,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray,
        dataset_val: np.ndarray | None,
        timesteps: np.ndarray,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader | None]:
        """
        Prepare the data for the predict or fit methods.
        Note: All datasets must have shape (N_samples, N_timesteps, N_chemicals).

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
        # Use batch size from the config if not provided
        if batch_size is None:
            batch_size = self.config.batch_size

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
        epochs: int | None = None,
    ) -> None:
        """
        Train the MultiONet model.

        Args:
            train_data (np.ndarray): The training data.
            test_data (np.ndarray): The test data (to evaluate the model during training).
            timesteps (np.ndarray): The timesteps.
            epochs (int, optional): The number of epochs to train the model.

        Returns:
            None
        """
        self.N_timesteps = len(timesteps)
        self.N_train_samples = int(len(train_loader.dataset) / self.N_timesteps)

        criterion = self.setup_criterion()
        optimizer, scheduler = self.setup_optimizer_and_scheduler()

        train_losses, test_losses, accuracies = self.setup_losses(epochs=epochs)

        epochs = self.config.num_epochs if epochs is None else epochs

        progress_bar = tqdm(range(epochs), desc="Training Progress")
        for epoch in progress_bar:
            train_losses[epoch] = self.epoch(train_loader, criterion, optimizer)

            clr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix({"loss": train_losses[epoch], "lr": clr})
            scheduler.step()

            if test_loader is not None:
                preds, targets = self.predict(
                    test_loader,
                    timesteps,
                )
                test_losses[epoch] = criterion(preds, targets).item() / torch.numel(
                    targets
                )
                accuracies[epoch] = 1.0 - torch.mean(
                    torch.abs(preds - targets) / torch.abs(targets)
                )

        self.train_loss = train_losses
        self.test_loss = test_losses
        self.accuracy = accuracies

    def predict(
        self,
        data_loader: DataLoader,
        timesteps: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the model on the test data.

        Args:
            data_loader (DataLoader): The DataLoader object containing the test data.
            N_timesteps (int): The number of timesteps.

        Returns:
            tuple: The predictions and targets.
        """
        N_timesteps = len(timesteps)
        device = self.device
        self.eval()
        self.to(device)

        dataset_size = len(data_loader.dataset)

        # Pre-allocate buffers for predictions and targets
        preds = torch.zeros((dataset_size, self.N), dtype=torch.float32, device=device)
        targets = torch.zeros(
            (dataset_size, self.N), dtype=torch.float32, device=device
        )

        start_idx = 0

        with torch.no_grad():
            for branch_inputs, trunk_inputs, batch_targets in data_loader:
                batch_size = branch_inputs.size(0)
                branch_inputs, trunk_inputs, batch_targets = (
                    branch_inputs.to(device),
                    trunk_inputs.to(device),
                    batch_targets.to(device),
                )
                outputs = self((branch_inputs, trunk_inputs, batch_targets))

                # Write predictions and targets to the pre-allocated buffers
                preds[start_idx : start_idx + batch_size] = outputs
                targets[start_idx : start_idx + batch_size] = batch_targets

                start_idx += batch_size

        preds = preds.reshape(-1, N_timesteps, self.N)
        targets = targets.reshape(-1, N_timesteps, self.N)

        return preds, targets

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
                self.config.masses, crit, weights, self.config.device
            )
        return crit

    def setup_optimizer_and_scheduler(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        Utility function to set up the optimizer and scheduler for training.

        Args:
            conf (dataclasses.dataclass): The configuration dataclass.
            deeponet (OperatorNetworkType): The model to train.

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
                optimizer,
                start_factor=1,
                end_factor=0.3,
                total_iters=self.config.num_epochs,
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1,
                end_factor=1,
                total_iters=self.config.num_epochs,
            )
        return optimizer, scheduler

    def setup_losses(
        self,
        prev_train_loss: Optional[np.ndarray] = None,
        prev_test_loss: Optional[np.ndarray] = None,
        prev_accuracy: Optional[np.ndarray] = None,
        epochs: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up the loss history arrays for training.

        Args:
            prev_train_loss (np.ndarray): Previous training loss history.
            prev_test_loss (np.ndarray): Previous test loss history.
            prev_accuracy (np.ndarray): Previous accuracy history.
            epochs (int, optional): The number of epochs to train the model.

        Returns:
            tuple: The training and testing loss history arrays (both np.ndarrays).
        """
        epochs = self.config.num_epochs if epochs is None else epochs
        if self.config.pretrained_model_path is None:
            train_losses = np.zeros(epochs)
            test_losses = np.zeros(epochs)
            accuracies = np.zeros(epochs)
        else:
            train_losses = np.concatenate((prev_train_loss, np.zeros(epochs)))
            test_losses = np.concatenate((prev_test_loss, np.zeros(epochs)))
            accuracies = np.concatenate((prev_accuracy, np.zeros(epochs)))

        return train_losses, test_losses, accuracies

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
            outputs = self((branch_input, trunk_input, targets))
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
        dataset = TensorDataset(
            branch_inputs_tensor, trunk_inputs_tensor, targets_tensor
        )

        def worker_init_fn(worker_id):
            torch_seed = torch.initial_seed()
            np_seed = torch_seed // 2**32 - 1
            np.random.seed(np_seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
            num_workers=4,
        )
