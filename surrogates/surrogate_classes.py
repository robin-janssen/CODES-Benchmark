from surrogates.DeepONet.deeponet import MultiONet
from surrogates.FCNN.fcnn import FullyConnected
from surrogates.NeuralODE.neural_ode import NeuralODE

# Define surrogate classes
surrogate_classes = [
    MultiONet,
    FullyConnected,
    NeuralODE,
    # Add any additional surrogate classes here
]
