import torch
import torch.nn as nn

# from torch.profiler import ProfilerActivity, record_function
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from surrogates.surrogates import AbstractSurrogateModel
from surrogates.FCNN.fcnn_config import OConfig

from utils import time_execution, worker_init_fn


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(FullyConnectedNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
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
    def __init__(self, device: str | None = None, N_chemicals: int = 29):
        """
        Initialize the FullyConnected model with a configuration.

        The configuration must provide the following information:
        - input_size (int): The input size for the network.
        - hidden_size (int): The number of hidden units in each layer of the network.
        - num_hidden_layers (int): The number of hidden layers in the network.
        - output_size (int): The number of outputs of the model.
        - device (str): The device to use for training (e.g., 'cpu', 'cuda:0').
        """
        config = OConfig()  # Load the specific config for FullyConnected
        super().__init__(device=device, N_chemicals=N_chemicals)

        self.config = config
        self.device = device
        self.N = N_chemicals
        self.model = FullyConnectedNet(
            self.N + 1,  # 29 chemicals + 1 time input
            config.hidden_size,
            self.N,
            config.num_hidden_layers,
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
        timesteps: np.ndarray,
        epochs: int,
        position: int = 0,
        description: str = "Training FullyConnected",
    ) -> None:
        """
        Train the FullyConnected model.

        Args:
            train_loader (DataLoader): The DataLoader object containing the training data.
            test_loader (DataLoader): The DataLoader object containing the test data.
            timesteps (np.ndarray): The timesteps.
            epochs (int, optional): The number of epochs to train the model.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.

        Returns:
            None
        """
        self.N_timesteps = len(timesteps)
        self.N_train_samples = int(len(train_loader.dataset) / self.N_timesteps)

        criterion = self.setup_criterion()
        optimizer, scheduler = self.setup_optimizer_and_scheduler(epochs)

        train_losses, test_losses, accuracies = (np.zeros(epochs),) * 3

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
                accuracies[epoch] = 1.0 - torch.mean(
                    torch.abs(preds - targets) / torch.abs(targets)
                )

        progress_bar.close()

        self.train_loss = train_losses
        self.test_loss = test_losses
        self.accuracy = accuracies

    def predict(
        self,
        data_loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the model on the test data.

        Args:
            data_loader (DataLoader): The DataLoader object containing the test data.
            timesteps (np.ndarray): The timesteps array.

        Returns:
            tuple: The total loss, outputs, and targets.
        """
        return super().predict(data_loader)

        # N_timesteps = len(timesteps)
        # device = self.device
        # self.eval()
        # self.to(device)

        # dataset_size = len(data_loader.dataset)

        # # Pre-allocate buffers for predictions and targets
        # preds = torch.zeros((dataset_size, self.N), dtype=torch.float32, device=device)
        # targets = torch.zeros(
        #     (dataset_size, self.N), dtype=torch.float32, device=device
        # )

        # start_idx = 0

        # with torch.no_grad():
        #     for inputs, batch_targets in data_loader:
        #         batch_size = inputs.size(0)
        #         inputs, batch_targets = (
        #             inputs.to(device),
        #             batch_targets.to(device),
        #         )
        #         outputs = self((inputs, batch_targets))

        #         # Write predictions and targets to the pre-allocated buffers
        #         preds[start_idx : start_idx + batch_size] = outputs
        #         targets[start_idx : start_idx + batch_size] = batch_targets

        #         start_idx += batch_size

        # preds = preds.reshape(-1, N_timesteps, self.N)
        # targets = targets.reshape(-1, N_timesteps, self.N)

        # preds = self.denormalize(preds)
        # targets = self.denormalize(targets)

        # return preds, targets

    def setup_criterion(self) -> callable:
        """
        Utility function to set up the loss function for training.

        Returns:
            callable: The loss function.
        """
        crit = nn.MSELoss(reduction="sum")
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
        normalize: bool = True,
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

        if normalize:
            inputs_tensor = (inputs_tensor - inputs_tensor.mean()) / inputs_tensor.std()
            targets_tensor = (
                targets_tensor - targets_tensor.mean()
            ) / targets_tensor.std()

        inputs_tensor = inputs_tensor.to(self.device)
        targets_tensor = targets_tensor.to(self.device)
        dataset = TensorDataset(inputs_tensor, targets_tensor)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
            # num_workers=4,
        )
