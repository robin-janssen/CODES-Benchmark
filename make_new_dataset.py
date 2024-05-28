import numpy as np
from data import create_hdf5_dataset

if __name__ == "__main__":
    # Create a new dataset
    train_data = np.load("data/osu2008/train_data.npy")
    test_data = np.load("data/osu2008/test_data.npy")
    create_hdf5_dataset(train_data, test_data, "osu2008")
