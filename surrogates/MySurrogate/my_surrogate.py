import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, abundances, device):
        # abundances with shape (n_samples, n_timesteps, n_species)
        self.abundances = torch.tensor(abundances, dtype=torch.float32).to(device)
        self.length = self.abundances.shape[0]

    def __getitem__(self, index):
        return self.abundances[index, :, :]

    # def __getitems__(self, index_list):  # for better batch performance, optional
    #     return self.abundances[index_list, :, :]

    def __len__(self):
        return self.length


from surrogates.surrogates import AbstractSurrogateModel
from torch import nn

from surrogates.MySurrogate.my_surrogate_config import MySurrogateConfig

from torch.utils.data import DataLoader
import numpy as np

from torch.optim import Adam
from utils import time_execution


class MySurrogate(AbstractSurrogateModel):

    def __init__(
        self,
        device: str | None,
        n_chemicals: int,
        n_timesteps: int,
        model_config: dict | None,
    ):
        super().__init__(device, n_chemicals, n_timesteps, model_config)

        model_config = model_config if model_config is not None else {}
        self.config = MySurrogateConfig(**model_config)

        # construct the model according to the parameters in the config
        modules = []
        modules.append(nn.Linear(n_chemicals, self.config.layer_width))
        modules.append(self.config.activation)
        for _ in range(self.config.hidden_layers):
            modules.append(nn.Linear(self.config.layer_width, self.config.layer_width))
            modules.append(self.config.activation)
        modules.append(nn.Linear(self.config.layer_width, n_chemicals*n_timesteps))

        self.model = nn.Sequential(*modules).to(device)

    def prepare_data(
        self,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray | None,
        dataset_val: np.ndarray | None,
        timesteps: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:

        train = MyDataset(dataset_train, self.device)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)

        if dataset_test is not None:
            test = MyDataset(dataset_test, self.device)
            test_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)
        else:
            test_loader = None

        if dataset_val is not None:
            val = MyDataset(dataset_val, self.device)
            val_loader = DataLoader(val, batch_size=batch_size, shuffle=shuffle)
        else:
            val_loader = None

        return train_loader, test_loader, val_loader

    def forward(self, inputs):
        targets = inputs
        initial_cond = inputs[..., 0, :]
        outputs = self.model(initial_cond)
        outputs = outputs.view(inputs.shape)
        return outputs, targets

    @time_execution
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        position: int,
        description: str,
    ):

        criterion = nn.MSELoss()
        optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

        # initialize the loss tensors
        losses = torch.empty((epochs, len(train_loader)))
        test_losses = torch.empty((epochs))
        MAEs = torch.empty((epochs))

        # setup the progress bar
        progress_bar = self.setup_progress_bar(epochs, position, description)

        # training loop as usual
        for epoch in progress_bar:
            for i, x_true in enumerate(train_loader):
                optimizer.zero_grad()
                x_pred, _ = self.forward(x_true)
                loss = criterion(x_true, x_pred)
                loss.backward()
                optimizer.step()
                losses[epoch, i] = loss.item()

            # set the progress bar output
            clr = optimizer.param_groups[0]["lr"]
            print_loss = f"{losses[epoch, -1].item():.2e}"
            progress_bar.set_postfix({"loss": print_loss, "lr": f"{clr:.1e}"})

            # evaluate the model on the test set
            with torch.inference_mode():
                self.model.eval()
                preds, targets = self.predict(test_loader)
                self.model.train()
                loss = criterion(preds, targets)
                test_losses[epoch] = loss
                MAEs[epoch] = self.L1(preds, targets).item()

        progress_bar.close()

        self.train_loss = torch.mean(losses, dim=1)
        self.test_loss = test_losses
        self.MAE = MAEs
