import numpy as np
from data import create_dataset

# from data.osu2008_old.osu_chemicals import osu_chemicals

if __name__ == "__main__":
    # Create a new dataset
    train_data = np.load("data/osu2008_old/train_data.npy")
    test_data = np.load("data/osu2008_old/test_data.npy")
    full_dataset = np.concatenate((train_data, test_data), axis=0)
    np.random.shuffle(full_dataset)
    labels = None
    create_dataset(
        "osu2008",
        full_dataset,
        split=(0.75, 0.05, 0.2),
        timesteps=np.linspace(0, 1, 100),
        labels=labels,
    )
