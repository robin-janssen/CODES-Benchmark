from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader_deeponet(
    data,
    timesteps,
    fraction=1,
    batch_size=32,
    shuffle=False,
    normalize=False,
):
    """
    Create a DataLoader with optional fractional subsampling for chemical evolution data for DeepONet.

    :param data: 3D numpy array with shape (num_samples, len(timesteps), num_chemicals)
    :param timesteps: 1D numpy array of timesteps.
    :param fraction: Fraction of the grid points to sample.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: Whether to shuffle the data.
    :param normalize: Whether to normalize the data.#
    :param device: Device to use.
    :return: A DataLoader object.
    """
    # Initialize lists to store the inputs and targets
    branch_inputs = []
    trunk_inputs = []
    targets = []

    # Iterate through the grid to select the samples
    if fraction == 1:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                branch_inputs.append(data[i, 0, :])
                trunk_inputs.append([timesteps[j]])
                targets.append(data[i, j, :])
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.random.uniform(0, 1) < fraction:
                    branch_inputs.append(data[i, :, 0])
                    trunk_inputs.append([timesteps[j]])
                    targets.append(data[i, :, j])

    # Convert to PyTorch tensors
    branch_inputs_tensor = torch.tensor(np.array(branch_inputs), dtype=torch.float32)
    trunk_inputs_tensor = torch.tensor(np.array(trunk_inputs), dtype=torch.float32)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

    if normalize:
        branch_inputs_tensor = (
            branch_inputs_tensor - branch_inputs_tensor.mean()
        ) / branch_inputs_tensor.std()
        trunk_inputs_tensor = (
            trunk_inputs_tensor - trunk_inputs_tensor.mean()
        ) / trunk_inputs_tensor.std()
        targets_tensor = (targets_tensor - targets_tensor.mean()) / targets_tensor.std()

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor, targets_tensor)

    def worker_init_fn(worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        np.random.seed(np_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=worker_init_fn,
        num_workers=4,
    )
