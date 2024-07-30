from surrogates.DeepONet.deeponet import MultiONet
from surrogates.FCNN.fcnn import FullyConnected
from surrogates.NeuralODE.neural_ode import NeuralODE
from surrogates.LatentPolynomial.latent_poly import LatentPolynomial

# Define surrogate classes
surrogate_classes = [
    MultiONet,
    FullyConnected,
    NeuralODE,
    LatentPolynomial,
    # Add any additional surrogate classes here
]
