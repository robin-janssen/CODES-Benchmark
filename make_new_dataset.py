import numpy as np
from data import create_hdf5_dataset

if __name__ == "__main__":
    # Create a new dataset
    train_data = np.load("data/osu2008/train_data.npy")
    test_data = np.load("data/osu2008/test_data.npy")
    full_dataset = np.concatenate((train_data, test_data), axis=0)
    np.random.shuffle(full_dataset)
    # Split the dataset into train, test, and validation sets
    # The train set is 75% of the full dataset
    train_data = full_dataset[: int(0.75 * len(full_dataset))]
    # The test set is only 5% of the full dataset because it is only used for loss trajectory plots
    test_data = full_dataset[
        int(0.75 * len(full_dataset)) : int(0.8 * len(full_dataset))
    ]
    # The validation set is 20% of the full dataset
    val_data = full_dataset[int(0.8 * len(full_dataset)) :]
    create_hdf5_dataset(train_data, test_data, val_data, "osu2008")
