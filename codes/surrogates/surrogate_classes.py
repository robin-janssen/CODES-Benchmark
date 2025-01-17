from .AbstractSurrogate import AbstractSurrogateModel

# Automatically retrieve all registered surrogate model classes
surrogate_classes = AbstractSurrogateModel.get_registered_classes()
