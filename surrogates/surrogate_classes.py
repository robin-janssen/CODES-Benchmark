from surrogates.DeepONet.deeponet import MultiONet
from surrogates.FCNN.fcnn import FullyConnected
from surrogates.LatentNeuralODE.neural_ode import LatentNeuralODE
from surrogates.LatentPolynomial.latent_poly import LatentPoly

# Define surrogate classes
surrogate_classes = [
    MultiONet,
    FullyConnected,
    LatentNeuralODE,
    LatentPoly,
    # Add any additional surrogate classes here
]
