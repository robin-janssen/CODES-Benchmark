import numpy as np
import optuna
import torch
import torch.nn as nn
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader, Dataset

from codes.surrogates.FCNN.fcnn_config import FCNNBaseConfig
from codes.surrogates.AbstractSurrogate.surrogates import AbstractSurrogateModel
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
    ) -> None:
        self.n_train_samples = int(len(train_loader.dataset) / self.n_timesteps)
        criterion = nn.MSELoss(reduction="sum")
        optimizer = self.setup_optimizer_and_scheduler()

        train_losses, test_losses, MAEs = [np.zeros(epochs) for _ in range(3)]
        progress_bar = self.setup_progress_bar(epochs, position, description)

        for epoch in progress_bar:
            train_losses[epoch] = self.epoch(train_loader, criterion, optimizer)

            clr = optimizer.param_groups[0]["lr"]
            print_loss = f"{train_losses[epoch].item():.2e}"
            progress_bar.set_postfix({"loss": print_loss, "lr": f"{clr:.1e}"})

            if test_loader is not None:
                self.eval()
                optimizer.eval()
                preds, targets = self.predict(test_loader)
                loss = criterion(preds, targets).item()
                loss /= len(test_loader.dataset) * self.N
                test_losses[epoch] = loss
                MAEs[epoch] = self.L1(preds, targets).item()

                if self.optuna_trial is not None:
                    self.optuna_trial.report(loss, epoch)
                    if self.optuna_trial.should_prune():
                        raise optuna.TrialPruned()

        progress_bar.close()
        self.n_epochs = epoch
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
        self.train()
        optimizer.train()
        total_loss = 0
        dataset_size = len(data_loader.dataset)

        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

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
        Create a DataLoader with precomputed batches to avoid overhead from slicing.

        Steps:
         1) Expand dataset shape to (samples, timesteps, chemicals+1) by adding time.
         2) Flatten to shape (samples*timesteps, chemicals+1) for FC input
            and (samples*timesteps, chemicals) for targets.
         3) Precompute batches of size [batch_size, chemicals+1] and [batch_size, chemicals].
         4) Store them in a FCPrebatchedDataset.
         5) Return a DataLoader with batch_size=1 and a custom collate_fn to remove extra dims.
        """
        n_samples, n_timesteps_, n_chemicals_ = dataset.shape
        # Make sure n_chemicals_ == self.N
        # Add 1 time dimension: shape = (samples, timesteps, chemicals+1)
        dataset_with_time = np.concatenate(
            [
                dataset,
                np.tile(timesteps, (n_samples, 1)).reshape(n_samples, n_timesteps_, 1),
            ],
            axis=2,
        )

        # Flatten data
        # shape -> (samples*timesteps, chemicals+1)
        flattened_inputs = dataset_with_time.reshape(-1, self.N + 1)
        # shape -> (samples*timesteps, chemicals)
        flattened_targets = dataset.reshape(-1, self.N)

        # Precompute batches
        batched_inputs = []
        batched_targets = []

        temp_inps, temp_tgts = [], []
        for i in range(flattened_inputs.shape[0]):
            temp_inps.append(flattened_inputs[i])
            temp_tgts.append(flattened_targets[i])

            if len(temp_inps) == batch_size:
                # Convert lists to arrays, then to tensors
                inp_tensor = torch.tensor(
                    np.array(temp_inps, dtype=np.float32), dtype=torch.float32
                )
                tgt_tensor = torch.tensor(
                    np.array(temp_tgts, dtype=np.float32), dtype=torch.float32
                )
                batched_inputs.append(inp_tensor)
                batched_targets.append(tgt_tensor)
                temp_inps, temp_tgts = [], []

        # Handle any leftovers
        if temp_inps:
            inp_tensor = torch.tensor(
                np.array(temp_inps, dtype=np.float32), dtype=torch.float32
            )
            tgt_tensor = torch.tensor(
                np.array(temp_tgts, dtype=np.float32), dtype=torch.float32
            )
            batched_inputs.append(inp_tensor)
            batched_targets.append(tgt_tensor)

        # Move everything to device
        batched_inputs = [b.to(self.device) for b in batched_inputs]
        batched_targets = [b.to(self.device) for b in batched_targets]

        # Create a pre-batched dataset
        dataset_prebatched = FCPrebatchedDataset(batched_inputs, batched_targets)

        # DataLoader returns batch_size=1, each "batch" is actually a full precomputed batch
        return DataLoader(
            dataset_prebatched,
            batch_size=1,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=fc_collate_fn,
            worker_init_fn=worker_init_fn,
        )


AbstractSurrogateModel.register(FullyConnected)
