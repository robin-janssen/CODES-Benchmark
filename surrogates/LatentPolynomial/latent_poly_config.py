from dataclasses import dataclass


@dataclass
class LatentPolynomialConfigOSU:
    "Model Config for LatentPolynomial model for OUS_2008 dataset"
    in_features: int = 29
    latent_features: int = 5
    degree: int = 2