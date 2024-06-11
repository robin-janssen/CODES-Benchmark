from surrogates.DeepONet.deeponet import MultiONet
from surrogates.FCNN.fcnn import FullyConnected

# Define surrogate classes
surrogate_classes = {
    "DeepONet": MultiONet,
    "FCNN": FullyConnected,
    # Add any additional surrogate classes here
}
