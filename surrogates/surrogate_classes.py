from surrogates.DeepONet.deeponet import MultiONet
from surrogates.FCNN.fcnn import FullyConnected
from surrogates.NeuralODE.neural_ode import NeuralODE

# Define surrogate classes
surrogate_classes = {
    "DeepONet": MultiONet,
    "FCNN": FullyConnected,
    "NeuralODE": NeuralODE,
    # Add any additional surrogate classes here
}
