# This is the main training script for the models. It trains the models and saves them.
# The structure of the file should be something like this:

# 1. Import the necessary libraries
# 2. Import the train functions for DeepONet and NeuralODE
# 3. Load the data
# 4. Train and save the models for the different purposes


# 1. Import the necessary libraries

import os
import numpy as np


# 2. Import the train functions for DeepONet and NeuralODE

from DeepONet.config_classes import OChemicalTrainConfig
from DeepONet.dataloader import create_dataloader_chemicals
from DeepONet.train_utils import save_model
from DeepONet.train_multionet import (
    train_multionet_chemical,
)


# 3. Load the data

data_path = "data/osu_data"
full_train_data = np.load(os.path.join(data_path, "train_data.npy"))
full_test_data = np.load(os.path.join(data_path, "test_data.npy"))
osu_timesteps = np.linspace(0, 99, 100)

print(f"Loaded data with shape: {full_train_data.shape}/{full_test_data.shape}")

# 4. Train and save the models for the different purposes

config = OChemicalTrainConfig()

# Main model for timing and accuracy

mod_train_data = full_train_data[:10]
mod_test_data = full_test_data[:10]

dataloader_train_full = create_dataloader_chemicals(
    mod_train_data, osu_timesteps, batch_size=config.batch_size, shuffle=True
)
dataloader_test_full = create_dataloader_chemicals(
    mod_test_data, osu_timesteps, batch_size=config.batch_size, shuffle=False
)
main_model, train_loss, test_loss = train_multionet_chemical(
    config, dataloader_train_full, dataloader_test_full
)

save_model(
    main_model,
    "main_deeponet.pth",
    config,
    train_loss,
    test_loss,
    train_multionet_chemical.duration,
)

# Models for interpolation testing

intervals = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

for interval in intervals:

    # Modify the data for interpolation testing
    train_data = full_train_data[:, ::interval]
    test_data = full_test_data[:, ::interval]

    dataloader_train = create_dataloader_chemicals(
        train_data,
        osu_timesteps[::interval],
        batch_size=config.batch_size,
        shuffle=True,
    )
    dataloader_test = create_dataloader_chemicals(
        test_data,
        osu_timesteps[::interval],
        batch_size=config.batch_size,
        shuffle=False,
    )

    model, train_loss, test_loss = train_multionet_chemical(
        config, dataloader_train, dataloader_test
    )

    save_model(
        model,
        f"deeponet_interpolation_{interval}.pth",
        config,
        train_loss,
        test_loss,
        train_multionet_chemical.duration,
    )

# Models for extrapolation testing

cutoffs = (50, 60, 70, 80, 90)

for cutoff in cutoffs:

    # Modify the data for extrapolation testing
    train_data = full_train_data[:, :cutoff]
    test_data = full_test_data[:, :cutoff]

    dataloader_train = create_dataloader_chemicals(
        train_data, osu_timesteps[:cutoff], batch_size=config.batch_size, shuffle=True
    )
    dataloader_test = create_dataloader_chemicals(
        test_data, osu_timesteps[:cutoff], batch_size=config.batch_size, shuffle=False
    )

    model, train_loss, test_loss = train_multionet_chemical(
        config, dataloader_train, dataloader_test
    )

    save_model(
        model,
        f"deeponet_extrapolation_{cutoff}.pth",
        config,
        train_loss,
        test_loss,
        train_multionet_chemical.duration,
    )

# Sparse data performance testing

factors = (2, 4, 8, 16, 32)

for factor in factors:

    # Modify the data for sparse data testing
    train_data = full_train_data[::factor]
    test_data = full_test_data[::factor]

    dataloader_train = create_dataloader_chemicals(
        train_data,
        osu_timesteps[::factor],
        batch_size=config.batch_size,
        shuffle=True,
    )
    dataloader_test = create_dataloader_chemicals(
        test_data,
        osu_timesteps[::factor],
        batch_size=config.batch_size,
        shuffle=False,
    )

    model, train_loss, test_loss = train_multionet_chemical(
        config, dataloader_train, dataloader_test
    )

    save_model(
        model,
        f"deeponet_sparse_{factor}.pth",
        config,
        train_loss,
        test_loss,
        train_multionet_chemical.duration,
    )
