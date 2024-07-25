import torch
from torch import nn

from surrogates.surrogates import AbstractSurrogateModel
from surrogates.LatentPolynomial.latent_poly_config import LatentPolynomialConfigOSU


class LatentPolynomial(AbstractSurrogateModel):

    def __init__(self, device: str | None = None):
        super().__init__()
        self.config: LatentPolynomialConfigOSU = LatentPolynomialConfigOSU()
        if device is not None:
            self.config.device = device


class Polynomial(nn.Module):
    """
    Polynomial class with learnable parameters

    Attributes:
        degree (int): the degree of the polynomial
        dimension (int): The dimension of the in- and output variables
    """

    def __init__(self, degree: int, dimension: int):
        super().__init__()
        self._coef = nn.Linear(in_features=degree, out_features=dimension, bias=False, dtype=torch.float32)
        self.degree = degree
        self.dimension = dimension
        self._t_matrix = None

    def forward(self, t: torch.Tensor):
        if self._t_matrix is None:
            self._t_matrix = self._prepare_t(t)
        return self._coef(self._t_matrix)

    def _prepare_t(self, t):
        t = t[:, None]
        return torch.hstack([t ** i for i in range(1, self.degree + 1)]).permute(0, 2, 1)
