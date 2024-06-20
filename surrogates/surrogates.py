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
        dataset: np.ndarray,
        timesteps: np.ndarray,
        batch_size: int | None,
        shuffle: bool,
    ):
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

    @abstractmethod
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

        print(f"Model, losses and hyperparameters saved to {model_dir}")