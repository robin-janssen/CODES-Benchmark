import sys

import numpy as np

sys.path.insert(1, "../..")

from codes import create_dataset

if __name__ == "__main__":
    # Create a new dataset
    dataset = np.random.rand(500, 101, 5)  # 500 samples, 100 timesteps, 5 features
    labels = [
        "label1",
        "label2",
        "label3",
        "label4",
        "label5",
    ]  # Labels should be in the same order as the features
    timesteps = np.linspace(
        0, 1, 101
    )  # Timesteps must have the same length as the second dimension of the dataset

    # The dataset will be split into 70% training, 10% validation, and 20% testing
    # The HDF5 file would be savd as datasets/new_dataset/data.hdf5
    create_dataset(
        "new_dataset",
        train_data=dataset,
        timesteps=timesteps,
        labels=labels,
        split=(0.7, 0.1, 0.2),
    )
