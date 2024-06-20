from abc import ABC, abstractmethod
import os
import dataclasses
import yaml

# from typing import Optional, Union

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import numpy as np

from utils import create_model_dir


# Define abstract base class for surrogate models
class AbstractSurrogateModel(ABC, nn.Module):

    def __init__(self):
        super().__init__()
        self.train_loss = None
        self.test_loss = None

    @abstractmethod
    def forward(self, inputs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        pass

    @abstractmethod
    def prepare_data(
        self,
        timesteps: np.ndarray,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray | None = None,
        dataset_val: np.ndarray | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        pass

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader | Tensor,
        test_loader: DataLoader | Tensor,
        timesteps: np.ndarray,
        epochs: int | None,
    ) -> None:
        pass

    @abstractmethod
    def predict(
        self,
        data_loader: DataLoader | Tensor,
        criterion: nn.Module,
        timesteps: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        pass

    def save(
        self,
        model_name: str,
        subfolder: str,
        training_id: str,
        dataset_name: str,
    ) -> None:

        base_dir = os.getcwd()
        subfolder = os.path.join(subfolder, training_id, self.__class__.__name__)
        model_dir = create_model_dir(base_dir, subfolder)

        # Save the model state dict
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(self.state_dict(), model_path)

        hyperparameters = dataclasses.asdict(self.config)
        hyperparameters["dataset_name"] = dataset_name
        for key in hyperparameters.keys():
            if isinstance(hyperparameters[key], nn.Module):
                hyperparameters[key] = hyperparameters[key].__class__.__name__

        hyperparameters_path = os.path.join(model_dir, f"{model_name}.yaml")
        with open(hyperparameters_path, "w") as file:
            yaml.dump(hyperparameters, file)

        # Save the losses as a numpy file
        if self.train_loss is None:
            self.train_loss = np.array([])
        if self.test_loss is None:
            self.test_loss = np.array([])
        losses_path = os.path.join(model_dir, f"{model_name}_losses.npz")
        np.savez(losses_path, train_loss=self.train_loss, test_loss=self.test_loss)

        print(f"Model, losses and hyperparameters saved to {model_dir}")

    def load(
            self, training_id: str, surr_name: str, model_identifier: str
        ) -> None:
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
