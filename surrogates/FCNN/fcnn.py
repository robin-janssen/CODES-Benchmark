# from torch.profiler import ProfilerActivity, record_function
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from surrogates.FCNN.fcnn_config import FCNNBaseConfig
from surrogates.surrogates import AbstractSurrogateModel
from utils import time_execution, worker_init_fn


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

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        One forward pass through the network.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the model.
        """
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

        The configuration must provide the following information:
        - hidden_size (int): The number of hidden units in each layer of the network.
        - num_hidden_layers (int): The number of hidden layers in the network.
        - learning_rate (float): The learning rate for the optimizer.
        - regularization_factor (float): The L2 regularization factor.
        - schedule (bool): Whether to use a learning rate schedule.
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
            self.N + 1,  # 29 chemicals + 1 time input
            self.config.hidden_size,
            self.N,
            self.config.num_hidden_layers,
            self.config.activation,
        ).to(device)

    def forward(
        self,
        inputs: tuple,
    ) -> torch.Tensor:
        """
        Forward pass for the FullyConnected model.

        Args:
            inputs (tuple[torch.Tensor, torch.Tensor]): The input tensor and the target tensor.
            Note: The targets are not used in the forward pass but are included for compatibility with the DataLoader.
            timesteps (np.ndarray, optional): The timesteps array.
            Note: The timesteps are not used in the forward pass but are included for compatibility with the benchmarking code.

        Returns:
            torch.Tensor: Output tensor of the model.
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

        # Create the DataLoaders
        dataloaders = []
        for dataset in [dataset_train, dataset_test, dataset_val]:
            if dataset is not None:
                dataloader = self.create_dataloader(
                    dataset, timesteps, batch_size, shuffle
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
        description: str = "Training FullyConnected",
    ) -> None:
        """
        Train the FullyConnected model.

        Args:
            train_loader (DataLoader): The DataLoader object containing the training data.
            test_loader (DataLoader): The DataLoader object containing the test data.
            epochs (int, optional): The number of epochs to train the model.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.

        Returns:
            None
        """
        # self.n_timesteps = len(timesteps)
        self.n_train_samples = int(len(train_loader.dataset) / self.n_timesteps)

        criterion = nn.MSELoss(reduction="sum")
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
                optimizer,
                start_factor=1,
                end_factor=0.3,
                total_iters=epochs,
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1,
                end_factor=1,
                total_iters=epochs,
            )
        return optimizer, scheduler

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
        total_loss = 0
        dataset_size = len(data_loader.dataset)

        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = (
                inputs.to(self.device),
                targets.to(self.device),
            )
            optimizer.zero_grad()
            outputs, targets = self.forward((inputs, targets))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= dataset_size * self.N
        return total_loss

    def create_dataloader(
        self,
        dataset: np.ndarray,
        timesteps: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
    ) -> DataLoader:
        """
        Create a DataLoader from a dataset.

        Args:
            dataset (np.ndarray): The dataset.
            timesteps (np.ndarray): The timesteps.
            batch_size (int): The batch size.
            shuffle (bool, optional): Whether to shuffle the data.

        Returns:
            DataLoader: The DataLoader object.
        """
        # Concatenate timesteps to the dataset as an additional feature
        n_samples, n_timesteps, n_features = dataset.shape
        dataset_with_time = np.concatenate(
            [
                dataset,
                np.tile(timesteps, (n_samples, 1)).reshape(n_samples, n_timesteps, 1),
            ],
            axis=2,
        )

        # Flatten the dataset for FCNN
        flattened_data = dataset_with_time.reshape(-1, n_features + 1)
        flattened_targets = dataset.reshape(-1, n_features)

        # Convert to PyTorch tensors
        inputs_tensor = torch.tensor(flattened_data, dtype=torch.float32)
        targets_tensor = torch.tensor(flattened_targets, dtype=torch.float32)

        inputs_tensor = inputs_tensor.to(self.device)
        targets_tensor = targets_tensor.to(self.device)
        dataset = TensorDataset(inputs_tensor, targets_tensor)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
        )
