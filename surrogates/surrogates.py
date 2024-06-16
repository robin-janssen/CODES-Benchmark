from abc import ABC, abstractmethod
from typing import Optional, Union

from torch import nn
from torch import Tensor


# Define abstract base class for surrogate models
class AbstractSurrogateModel(ABC, nn.Module):

    train_loss: Union[None, list[float], Tensor]
    test_loss: Union[None, list[float], Tensor]

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def prepare_data(self, data_loader):
        pass

    @abstractmethod
    def fit(self, conf, data_loader, test_loader=None):
        pass

    @abstractmethod
    def predict(self, data_loader):
        pass

    @abstractmethod
    def save(
        self, model_name, config, subfolder, train_loss, test_loss, training_duration
    ):
        pass
