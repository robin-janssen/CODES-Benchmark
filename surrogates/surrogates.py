from abc import ABC, abstractmethod

# from typing import Optional, Union

from torch import nn, Tensor
from torch.utils.data import DataLoader
import numpy as np


# Define abstract base class for surrogate models
class AbstractSurrogateModel(ABC, nn.Module):

    def __init__(self):
        super().__init__()

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
        pass
