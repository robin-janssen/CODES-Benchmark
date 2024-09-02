from surrogates.DeepONet.deeponet import MultiONet
from surrogates.FCNN.fcnn import FullyConnected
from surrogates.LatentNeuralODE.latent_neural_ode import LatentNeuralODE
from surrogates.LatentPolynomial.latent_poly import LatentPoly
from surrogates.MySurrogate.my_surrogate import MySurrogate

surrogate_classes = [
    MultiONet,
    FullyConnected,
    LatentNeuralODE,
    LatentPoly,
    # Add any additional surrogate classes here
    MySurrogate,
]
