import numpy as np
import optuna
import torch
import torch.nn as nn
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader, Dataset

from codes.surrogates.AbstractSurrogate.surrogates import AbstractSurrogateModel
from codes.surrogates.FCNN.fcnn_config import FCNNBaseConfig
from codes.utils import time_execution, worker_init_fn


def fc_collate_fn(batch):
    """
    Custom collate function to ensure tensors are returned in the correct shape.
    Args:
        batch: A list of tuples (input_batch, target_batch)
               where each item is already a precomputed batch of shape:
                 input_batch -> [batch_size, n_chemicals+1]
                 target_batch -> [batch_size, n_chemicals]
    Returns:
        A tuple of tensors with the final shapes:
        - inputs: [batch_size, n_chemicals+1]
        - targets: [batch_size, n_chemicals]
    """
    # 'batch' is a list of length=1 if DataLoader has batch_size=1,
    # and each element is (input_tensor, target_tensor).
    # We remove the extra [1,...,...] dimension via squeeze(0).
    inputs = torch.stack([item[0] for item in batch]).squeeze(0)
    targets = torch.stack([item[1] for item in batch]).squeeze(0)
    return inputs, targets


class FCPrebatchedDataset(Dataset):
    """
    Dataset for pre-batched data specifically for the FullyConnected model.
    Args:
        inputs_batches (list[Tensor]): List of precomputed input batches.
        targets_batches (list[Tensor]): List of precomputed target batches.
    """

    def __init__(self, inputs_batches, targets_batches):
        self.inputs_batches = inputs_batches
        self.targets_batches = targets_batches

    def __getitem__(self, index):
        # Return one precomputed batch:
        # shape -> ( [batch_size, n_chemicals+1], [batch_size, n_chemicals] )
        return self.inputs_batches[index], self.targets_batches[index]

    def __len__(self):
        # Number of precomputed batches
        return len(self.inputs_batches)


class FullyConnectedNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_hidden_layers,
        activation=nn.ReLU(),
    ):
        super(FullyConnectedNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), activation]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), activation]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class FullyConnected(AbstractSurrogateModel):
    def __init__(
        self,
        device: str | None = None,
        n_chemicals: int = 29,
        n_timesteps: int = 100,
        config: dict | None = None,
    ):
        """
        Initialize the FullyConnected model with a configuration.

        The configuration must provide the following keys:
        - hidden_size (int)
        - num_hidden_layers (int)
        - learning_rate (float)
        - regularization_factor (float)
        - schedule (bool)
        - activation (nn.Module name or instance)
        """
        super().__init__(
            device=device,
            n_chemicals=n_chemicals,
            n_timesteps=n_timesteps,
            config=config,
        )
        self.config = FCNNBaseConfig(**self.config)
        self.device = device
        self.N = n_chemicals

        self.model = FullyConnectedNet(
            input_size=self.N + 1,  # 29 chemicals + 1 time input
            hidden_size=self.config.hidden_size,
            output_size=self.N,
            num_hidden_layers=self.config.num_hidden_layers,
            activation=self.config.activation,
        ).to(device)

    def forward(self, inputs: tuple) -> torch.Tensor:
        """
        Forward pass for the FullyConnected model.

        Args:
            inputs (tuple[torch.Tensor, torch.Tensor]):
                (x, targets) - 'targets' is included for a consistent interface
        Returns:
            (outputs, targets)
        """
        x, targets = inputs
        return self.model(x), targets

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
        All datasets: shape (n_samples, n_timesteps, n_chemicals)

        Returns: train_loader, test_loader, val_loader
        """
        dataloaders = []
        loader = self.create_dataloader(dataset_train, timesteps, batch_size, shuffle)
        dataloaders.append(loader)
        for dataset in [dataset_test, dataset_val]:
            if dataset is not None:
                loader = self.create_dataloader(
                    dataset, timesteps, batch_size, shuffle=False
                )
                dataloaders.append(loader)
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
        description: str = "Training FullyConnected",
        multi_objective: bool = False,
    ) -> None:
        """
        Train the FullyConnected model.

        Args:
            train_loader (DataLoader): The DataLoader object containing the training data.
            test_loader (DataLoader): The DataLoader object containing the test data.
            epochs (int, optional): The number of epochs to train the model.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.
            multi_objective (bool): Whether multi-objective optimization is used.
                                    If True, trial.report is not used (not supported by Optuna).

        Returns:
            None. The training loss, test loss, and MAE are stored in the model.
        """
        self.n_train_samples = int(len(train_loader.dataset) / self.n_timesteps)
        # criterion = nn.MSELoss(reduction="sum")
        criterion = nn.MSELoss()
        optimizer = self.setup_optimizer_and_scheduler()

        loss_length = (epochs + self.update_epochs - 1) // self.update_epochs
        train_losses, test_losses, MAEs = [np.zeros(loss_length) for _ in range(3)]
        progress_bar = self.setup_progress_bar(epochs, position, description)

        self.train()
        optimizer.train()

        for epoch in progress_bar:
            self.epoch(train_loader, criterion, optimizer)

            if epoch % self.update_epochs == 0:
                index = epoch // self.update_epochs
                # Set model and optimizer to eval mode for evaluation
                self.eval()
                optimizer.eval()

                # Calculate losses and MAE
                preds, targets = self.predict(train_loader)
                train_losses[index] = criterion(preds, targets).item()
                preds, targets = self.predict(test_loader)
                test_losses[index] = criterion(preds, targets).item()
                MAEs[index] = self.L1(preds, targets).item()

                # Update progress bar postfix
                postfix = {
                    "train_loss": f"{train_losses[index]:.2e}",
                    "test_loss": f"{test_losses[index]:.2e}"
                }
                progress_bar.set_postfix(postfix)

                # Report the test loss to Optuna
                if self.optuna_trial is not None and not multi_objective:
                    self.optuna_trial.report(test_losses[index], step=epoch)
                    if self.optuna_trial.should_prune():
                        raise optuna.TrialPruned()

                self.train()
                optimizer.train()

        progress_bar.close()
        self.n_epochs = epoch + 1
        self.train_loss = train_losses
        self.test_loss = test_losses
        self.MAE = MAEs

    def setup_optimizer_and_scheduler(self) -> torch.optim.Optimizer:
        """
        Utility function to set up the optimizer and (optionally) scheduler.
        """
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
        # dataset_size = len(data_loader.dataset)

        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs, targets = self.forward((inputs, targets))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    def create_dataloader(
        self,
        dataset: np.ndarray,
        timesteps: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
    ) -> DataLoader:
        """
        Create a DataLoader with optimized memory-safe shuffling and batching.

        Args:
            dataset (np.ndarray): The data to load. Shape: (n_samples, n_timesteps, n_chemicals).
            timesteps (np.ndarray): The timesteps. Shape: (n_timesteps,).
            batch_size (int): The batch size.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

        Returns:
            DataLoader: A DataLoader with precomputed batches.
        """
        device = self.device
        n_samples, n_timesteps, n_chemicals = dataset.shape
        total_samples = n_samples * n_timesteps

        # Expand dataset with time as an additional feature
        tiled_timesteps = np.tile(timesteps, (n_samples, 1))
        dataset_with_time = np.concatenate(
            [dataset, tiled_timesteps.reshape(n_samples, n_timesteps, 1)], axis=2
        )

        # Flatten the data
        flattened_inputs = dataset_with_time.reshape(-1, n_chemicals + 1)
        flattened_targets = dataset.reshape(-1, n_chemicals)

        if shuffle:
            permutation = np.random.permutation(total_samples)
            flattened_inputs = flattened_inputs[permutation]
            flattened_targets = flattened_targets[permutation]

        # Slice the flattened data into batches
        num_full_batches = total_samples // batch_size
        remainder = total_samples % batch_size

        batched_inputs = []
        batched_targets = []

        for batch_idx in range(num_full_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            batch_input = (
                torch.from_numpy(flattened_inputs[start:end]).float().to(device)
            )
            batch_target = (
                torch.from_numpy(flattened_targets[start:end]).float().to(device)
            )

            batched_inputs.append(batch_input)
            batched_targets.append(batch_target)

        if remainder > 0:
            start = num_full_batches * batch_size
            batch_input = torch.from_numpy(flattened_inputs[start:]).float().to(device)
            batch_target = (
                torch.from_numpy(flattened_targets[start:]).float().to(device)
            )

            batched_inputs.append(batch_input)
            batched_targets.append(batch_target)

        # Create the pre-batched dataset
        dataset_prebatched = FCPrebatchedDataset(batched_inputs, batched_targets)

        # Create the DataLoader with the original configuration
        return DataLoader(
            dataset_prebatched,
            batch_size=1,  # Each "batch" is a precomputed batch
            shuffle=False,  # Shuffle the order of batches each epoch
            num_workers=0,
            collate_fn=fc_collate_fn,
            worker_init_fn=worker_init_fn,
        )


AbstractSurrogateModel.register(FullyConnected)
