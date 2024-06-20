import dataclasses
import os
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, record_function
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from typing import Tuple, Optional
import yaml

from surrogates.surrogates import AbstractSurrogateModel
from surrogates.FCNN.fcnn_config import OConfig

from utils import time_execution, create_model_dir


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(FullyConnectedNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FullyConnected(AbstractSurrogateModel):
    def __init__(self, device: str = None):
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
        super(FullyConnected, self).__init__()

        self.config = config
        if device is not None:
            config.device = device
        self.device = config.device
        self.model = FullyConnectedNet(
            config.input_size,
            config.hidden_size,
            config.output_size,
            config.num_hidden_layers,
        ).to(config.device)

    def forward(self, inputs: Tuple) -> torch.Tensor:
        """
        Forward pass for the FullyConnected model.

        Args:
            inputs: Tuple containing the input tensor and the target tensor.
            Note: The targets are not used in the forward pass but are included for compatibility with the DataLoader.

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        x, _ = inputs
        return self.model(x)

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
        batch_size = self.config.batch_size if batch_size is None else batch_size

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

        # Create a TensorDataset and DataLoader
        # dataset = TensorDataset(
        #     inputs_tensor.to(self.device), targets_tensor.to(self.device)
        # )
        dataset = TensorDataset(inputs_tensor, targets_tensor)

        def worker_init_fn(worker_id):
            torch_seed = torch.initial_seed()
            np_seed = torch_seed // 2**32 - 1
            np.random.seed(np_seed)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
            num_workers=4,
        )

        self.dataset_size = len(dataloader.dataset)

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
        Train the FullyConnected model.

        Args:
            train_loader (DataLoader): The DataLoader object containing the training data.
            test_loader (DataLoader): The DataLoader object containing the test data.
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
            timesteps (np.ndarray): The timesteps array.

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
        preds = torch.zeros(
            (dataset_size, self.config.output_size), dtype=torch.float32, device=device
        )
        targets = torch.zeros(
            (dataset_size, self.config.output_size), dtype=torch.float32, device=device
        )

        start_idx = 0

        with torch.no_grad():
            for inputs, batch_targets in data_loader:
                batch_size = inputs.size(0)
                inputs, batch_targets = (
                    inputs.to(device),
                    batch_targets.to(device),
                )
                outputs = self((inputs, batch_targets))
                loss = criterion(outputs, batch_targets)
                total_loss += loss.item()

                # Write predictions and targets to the pre-allocated buffers
                preds[start_idx : start_idx + batch_size] = outputs
                targets[start_idx : start_idx + batch_size] = batch_targets

                start_idx += batch_size

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        # Calculate relative error
        total_loss /= dataset_size * self.config.output_size

        preds = preds.reshape(-1, N_timesteps, self.config.output_size)
        targets = targets.reshape(-1, N_timesteps, self.config.output_size)

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
        subfolder = os.path.join(subfolder, training_id, "FCNN")
        model_dir = create_model_dir(base_dir, subfolder)
        self.dataset_name = dataset_name

        # Save the model state dict
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(self.state_dict(), model_path)

        # Create the hyperparameters dictionary from the config dataclass
        hyperparameters = dataclasses.asdict(self.config)

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
        return crit

    def setup_optimizer_and_scheduler(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        Utility function to set up the optimizer and scheduler for training.

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
        N_outputs = self.config.output_size
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = (
                inputs.to(self.device),
                targets.to(self.device),
            )
            optimizer.zero_grad()
            outputs = self.forward((inputs, targets))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_loss /= dataset_size * N_outputs
        return total_loss

    def epoch_profiled(
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
        N_outputs = self.config.output_size

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/HTA2"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        ) as prof:
            i = 0
            for batch in data_loader:
                i += 1
                if i >= 30:
                    break
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                with record_function("model_inference"):
                    outputs = self.forward((inputs, targets))
                torch.cuda.synchronize()

                with record_function("loss_calculation"):
                    loss = criterion(outputs, targets)

                with record_function("backward_pass"):
                    loss.backward()
                torch.cuda.synchronize()

                with record_function("optimizer_step"):
                    optimizer.step()
                torch.cuda.synchronize()

                total_loss += loss.item()

                prof.step()

        total_loss /= self.dataset_size * N_outputs
        return total_loss, prof