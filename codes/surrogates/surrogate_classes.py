from codes.surrogates.DeepONet.deeponet import MultiONet
from codes.surrogates.FCNN.fcnn import FullyConnected
from codes.surrogates.LatentNeuralODE.latent_neural_ode import LatentNeuralODE
from codes.surrogates.LatentPolynomial.latent_poly import LatentPoly

surrogate_classes = [
    MultiONet,
    FullyConnected,
    LatentNeuralODE,
    LatentPoly,
    # Add any additional surrogate classes here
]
