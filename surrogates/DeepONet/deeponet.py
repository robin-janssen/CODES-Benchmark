import dataclasses
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Optional, TypeVar
import yaml

from surrogates.surrogates import AbstractSurrogateModel

# Use the below import to adjust the config class to the specific model
from surrogates.DeepONet.deeponet_config import OChemicalTrainConfig as MultiONetConfig
from data import create_dataloader_deeponet

from utils import time_execution, create_model_dir
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
    def __init__(self, device: str = None):
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
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass for the MultiONet model.

        Args:
            inputs (tuple): The input tuple containing branch_input, trunk_input, and targets.
            Note: The targets are not used in the forward pass, but are included for compatibility with DataLoader.

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
        dataset: np.ndarray,
        timesteps: np.ndarray,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Prepare the data for the predict or fit methods.

        Args:
            dataset (np.ndarray): The dataset to prepare (should be of shape (n_samples, n_timesteps, n_features)).
            timesteps (np.ndarray): The timesteps.
            batch_size (int, optional): The batch size.
            shuffle (bool, optional): Whether to shuffle the data.

        Returns:
            dataloader: The DataLoader object containing the prepared data.
        """
        # Use batch size from the config if not provided
        if batch_size is None:
            batch_size = self.config.batch_size

        # Create the DataLoader object
        dataloader = create_dataloader_deeponet(
            dataset,
            timesteps,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return dataloader

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

        train_loss_hist, test_loss_hist = self.setup_losses(
            prev_train_loss=None, prev_test_loss=None, epochs=epochs
        )

        epochs = self.config.num_epochs if epochs is None else epochs

        progress_bar = tqdm(range(epochs), desc="Training Progress")
        for epoch in progress_bar:
            train_loss_hist[epoch] = self.epoch(train_loader, criterion, optimizer)

            clr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix({"loss": train_loss_hist[epoch], "lr": clr})
            scheduler.step()

            if test_loader is not None:
                test_loss_hist[epoch], _, _ = self.predict(
                    test_loader,
                    criterion,
                    timesteps,
                )

        self.train_loss = train_loss_hist
        self.test_loss = test_loss_hist

    def predict(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        timesteps: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate the model on the test data.

        Args:
            data_loader (DataLoader): The DataLoader object containing the test data.
            criterion (nn.Module): The loss function.
            N_timesteps (int): The number of timesteps.

        Returns:
            tuple: The total loss, outputs, and targets.
        """
        N_timesteps = len(timesteps)
        device = self.device
        self.eval()
        self.to(device)

        total_loss = 0
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
                loss = criterion(outputs, batch_targets)
                total_loss += loss.item()

                # Write predictions and targets to the pre-allocated buffers
                preds[start_idx : start_idx + batch_size] = outputs
                targets[start_idx : start_idx + batch_size] = batch_targets

                start_idx += batch_size

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        # Calculate relative error
        total_loss /= dataset_size * self.N

        preds = preds.reshape(-1, N_timesteps, self.N)
        targets = targets.reshape(-1, N_timesteps, self.N)

        return total_loss, preds, targets

    def save(
        self,
        model_name: str,
        subfolder: str = "trained",
        training_id: str = "run_1",
        dataset_name: str = "dataset",
    ) -> None:
        """
        Save the trained model and hyperparameters.

        Args:
            model_name (str): The name of the model.
            subfolder (str): The subfolder to save the model in.
            training_id (str): A unique identifier to include in the directory name.
            dataset_name (str): The name of the dataset.
        """
        base_dir = os.getcwd()
        subfolder = os.path.join(subfolder, training_id, "DeepONet")
        model_dir = create_model_dir(base_dir, subfolder)
        self.dataset_name = dataset_name

        # Save the model state dict
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(self.state_dict(), model_path)

        # Create the hyperparameters dictionary from the config dataclass
        hyperparameters = dataclasses.asdict(self.config)

        # Remove the masses list from the hyperparameters
        hyperparameters.pop("masses", None)

        # Append the train time to the hyperparameters
        hyperparameters["train_duration"] = self.fit.duration
        hyperparameters["N_train_samples"] = self.N_train_samples
        hyperparameters["N_timesteps"] = self.N_timesteps
        hyperparameters["dataset_name"] = self.dataset_name

        # Save hyperparameters as a YAML file
        hyperparameters_path = os.path.join(model_dir, f"{model_name}.yaml")
        with open(hyperparameters_path, "w") as file:
            yaml.dump(hyperparameters, file)

        if self.train_loss is not None and self.test_loss is not None:
            # Save the losses as a numpy file
            losses_path = os.path.join(model_dir, f"{model_name}_losses.npz")
            np.savez(losses_path, train_loss=self.train_loss, test_loss=self.test_loss)

        print(f"Model, losses and hyperparameters saved to {model_dir}")

    def load(
        self, training_id: str, surr_name: str, model_identifier: str
    ) -> torch.nn.Module:
        """
        Load a trained surrogate model.

        Args:
            model: Instance of the surrogate model class.
            training_id (str): The training identifier.
            surr_name (str): The name of the surrogate model.
            model_identifier (str): The identifier of the model (e.g., 'main').

        Returns:
            The loaded surrogate model.
        """
        statedict_path = os.path.join(
            "trained", training_id, surr_name, f"{model_identifier}.pth"
        )
        self.load_state_dict(torch.load(statedict_path))
        self.eval()

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
        epochs: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up the loss history arrays for training.

        Args:
            prev_train_loss (np.ndarray): Previous training loss history.
            prev_test_loss (np.ndarray): Previous test loss history.
            epochs (int, optional): The number of epochs to train the model.

        Returns:
            tuple: The training and testing loss history arrays (both np.ndarrays).
        """
        epochs = self.config.num_epochs if epochs is None else epochs
        if self.config.pretrained_model_path is None:
            train_loss_hist = np.zeros(epochs)
            test_loss_hist = np.zeros(epochs)
        else:
            train_loss_hist = np.concatenate((prev_train_loss, np.zeros(epochs)))
            test_loss_hist = np.concatenate((prev_test_loss, np.zeros(epochs)))

        return train_loss_hist, test_loss_hist

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
        N_outputs = self.config.N_outputs

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

        total_loss /= dataset_size * N_outputs
        return total_loss
