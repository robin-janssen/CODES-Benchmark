import pytest
import random
import string

import torch

from surrogates.LatentPolynomial.latent_poly import LatentPoly
from surrogates.LatentNeuralODE.latent_neural_ode import LatentNeuralODE
from surrogates.DeepONet.deeponet import MultiONet


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_CHEMICALS = 10
N_TIMESTEPS = 50

classes = [LatentPoly, LatentNeuralODE]


@pytest.fixture(params=classes)
def instance(request):
    return request.param(DEVICE, N_CHEMICALS, N_TIMESTEPS)


@pytest.fixture
def dataloaders(instance):
    data_train = torch.rand((3, N_TIMESTEPS, N_CHEMICALS), dtype=torch.float64)
    data_test = torch.rand((3, N_TIMESTEPS, N_CHEMICALS), dtype=torch.float64)
    data_val = torch.rand((3, N_TIMESTEPS, N_CHEMICALS), dtype=torch.float64)
    timesteps = torch.rand(N_TIMESTEPS, dtype=torch.float64)
    batch_size = 1
    shuffle = True
    dataloader_train, dataloader_test, dataloader_val = instance.prepare_data(
        data_train, data_test, data_val, timesteps, batch_size, shuffle
    )
    return dataloader_train, dataloader_test, dataloader_val


def test_dataloaders(dataloaders):
    dataloader_train, dataloader_test, dataloader_val = dataloaders

    # Number of batches
    assert len(dataloader_train) == 3
    assert len(dataloader_test) == 3
    assert len(dataloader_val) == 3

    # Data shape
    assert next(iter(dataloader_train))[0].shape == torch.Size(
        [1, N_TIMESTEPS, N_CHEMICALS]
    )
    assert next(iter(dataloader_test))[0].shape == torch.Size(
        [1, N_TIMESTEPS, N_CHEMICALS]
    )
    assert next(iter(dataloader_val))[0].shape == torch.Size(
        [1, N_TIMESTEPS, N_CHEMICALS]
    )

    # Timesteps shape
    assert next(iter(dataloader_train))[1].shape == torch.Size([N_TIMESTEPS])
    assert next(iter(dataloader_test))[1].shape == torch.Size([N_TIMESTEPS])
    assert next(iter(dataloader_val))[1].shape == torch.Size([N_TIMESTEPS])


def test_forward(instance, dataloaders):
    dataloader_train, _, _ = dataloaders
    predictions, targets = instance.forward(next(iter(dataloader_train)))

    assert predictions.shape == torch.Size([1, N_TIMESTEPS, N_CHEMICALS])
    assert targets.shape == torch.Size([1, N_TIMESTEPS, N_CHEMICALS])


def test_predict(instance, dataloaders):
    dataloader_train, _, _ = dataloaders
    predictions, targets = instance.predict(dataloader_train)

    assert predictions.shape == torch.Size([3, N_TIMESTEPS, N_CHEMICALS])
    assert targets.shape == torch.Size([3, N_TIMESTEPS, N_CHEMICALS])


def test_fit(instance, dataloaders):
    dataloader_train, dataloader_test, _ = dataloaders
    instance.fit(dataloader_train, dataloader_test, epochs=2)

    assert instance.train_loss.shape == torch.Size([2])
    assert instance.test_loss.shape == torch.Size([2])
    assert instance.MAE.shape == torch.Size([2])


def test_save_load(instance, tmp_path):
    model_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    instance.save(
        model_name=model_name, base_dir=tmp_path, training_id="TestID", data_params={}
    )
    new_instance = instance.__class__(DEVICE, N_CHEMICALS, N_TIMESTEPS)
    new_instance.load(
        training_id="TestID",
        surr_name=instance.__class__.__name__,
        model_identifier=model_name,
        model_dir=tmp_path,
    )
