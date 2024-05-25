from abc import ABC, abstractmethod
import torch.nn as nn
from surrogates.DeepONet.deeponet import MultiONet

# Define surrogate classes
surrogate_classes = {
    "DeepONet": MultiONet,
    # Add any additional surrogate classes here
}


# Define abstract base class for surrogate models
class AbstractSurrogateModel(ABC, nn.Module):
    def __init__(self):
        super(AbstractSurrogateModel, self).__init__()

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
