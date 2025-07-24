import random
import string

import numpy as np
import pytest
import torch

from codes import surrogate_classes

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_CHEMICALS = 10
N_TIMESTEPS = 50

BATCH_SIZE = 1


@pytest.fixture(params=surrogate_classes)
def instance(request):
    return request.param(DEVICE, N_CHEMICALS, N_TIMESTEPS)


@pytest.fixture
def dataloaders(instance):
    data_train = np.random.rand(3, N_TIMESTEPS, N_CHEMICALS)
    data_test = np.random.rand(3, N_TIMESTEPS, N_CHEMICALS)
    data_val = np.random.rand(3, N_TIMESTEPS, N_CHEMICALS)
    timesteps = np.linspace(0, 1, N_TIMESTEPS)
    shuffle = True
    dataloader_train, dataloader_test, dataloader_val = instance.prepare_data(
        data_train, data_test, data_val, timesteps, BATCH_SIZE, shuffle
    )
    return dataloader_train, dataloader_test, dataloader_val


def test_init(instance):
    assert instance.device == DEVICE, f"device is wrong: {instance.device} != {DEVICE}"
    assert (
        instance.n_quantities == N_CHEMICALS
    ), f"n_quantities is wrong: {instance.n_quantities} != {N_CHEMICALS}"
    assert (
        instance.n_timesteps == N_TIMESTEPS
    ), f"n_timesteps is wrong: {instance.n_timesteps} != {N_TIMESTEPS}"
    assert instance.config is not None, "config is None"


def test_dataloader(instance, dataloaders):
    dataloader_train, dataloader_test, dataloader_val = dataloaders

    assert isinstance(
        dataloader_train, torch.utils.data.DataLoader
    ), "dataloader_train is not a DataLoader"
    assert isinstance(
        dataloader_test, torch.utils.data.DataLoader
    ), "dataloader_test is not a DataLoader"
    assert isinstance(
        dataloader_val, torch.utils.data.DataLoader
    ), "dataloader_val is not a DataLoader"

    # Get a batch from each dataloader and check its size
    train_batch = next(iter(dataloader_train))
    test_batch = next(iter(dataloader_test))
    val_batch = next(iter(dataloader_val))

    assert (
        train_batch[0].size(0) == BATCH_SIZE
    ), f"train batch has wrong size: {train_batch[0].size(0)} != {BATCH_SIZE}"
    assert (
        test_batch[0].size(0) == BATCH_SIZE
    ), f"test batch has wrong size: {test_batch[0].size(0)} != {BATCH_SIZE}"
    assert (
        val_batch[0].size(0) == BATCH_SIZE
    ), f"validation batch has wrong size: {val_batch[0].size(0)} != {BATCH_SIZE}"


def test_predict(instance, dataloaders):
    dataloader_train, _, _ = dataloaders
    predictions, targets = instance.predict(dataloader_train)

    assert predictions.shape == torch.Size(
        [3, N_TIMESTEPS, N_CHEMICALS]
    ), f"predictions has wrong shape: {predictions.shape} != [3, {N_TIMESTEPS}, {N_CHEMICALS}]"
    assert targets.shape == torch.Size(
        [3, N_TIMESTEPS, N_CHEMICALS]
    ), f"targets has wrong shape: {targets.shape} != [3, {N_TIMESTEPS}, {N_CHEMICALS}]"


def test_fit(instance, dataloaders):
    dataloader_train, dataloader_test, _ = dataloaders
    instance.fit(dataloader_train, dataloader_test, epochs=2)

    assert instance.train_loss.shape == torch.Size(
        [1]
    ), f"train loss has wrong shape: {instance.train_loss.shape} != [1]"
    assert instance.test_loss.shape == torch.Size(
        [1]
    ), f"test loss has wrong shape: {instance.test_loss.shape} != [1]"
    assert instance.MAE.shape == torch.Size(
        [1]
    ), f"MAE has wrong shape: {instance.MAE.shape} != [1]"


def test_save_load(instance, tmp_path):
    model_name = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
    )
    instance.save(model_name=model_name, base_dir=tmp_path, training_id="TestID")
    new_instance = instance.__class__(DEVICE, N_CHEMICALS, N_TIMESTEPS)
    new_instance.load(
        training_id="TestID",
        surr_name=instance.__class__.__name__,
        model_identifier=model_name,
        model_dir=tmp_path,
    )
    assert new_instance is not None, "model is None after loading"
