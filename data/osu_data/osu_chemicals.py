import os
import numpy as np

from models.DeepONet.utils import get_project_path

# from DeepONet.data_utils import train_test_split


def load_chemical_data(data_folder, file_extension=".dat", separator=" "):
    """
    Load chemical data from a directory containing multiple files.

    :param data_folder: The directory containing the data files.
    :param file_extension: The file extension of the data files.
    :return: A list of numpy arrays containing the data from each file.
    """
    # Get a list of all relevant files in the directory
    dataset_path = get_project_path(data_folder)
    all_files = os.listdir(dataset_path)
    files = [file for file in all_files if file.endswith(file_extension)]
    num_files = len(files)
    files.sort()

    # Load one file to see the data shape
    data = np.loadtxt(os.path.join(dataset_path, files[0]), delimiter=separator)
    data_shape = data.shape

    # Create an array to store all the data
    all_data = np.zeros((num_files, *data_shape))

    # Iterate over all the files and load the data
    for i, file in enumerate(files):
        if file.endswith(file_extension):
            data = np.loadtxt(os.path.join(dataset_path, file), delimiter=separator)
            all_data[i] = data

    return all_data


osu_chemicals = "H, H+, H2, H2+, H3+, O, O+, OH+, OH, O2, O2+, H2O, H2O+, H3O+, C, C+, CH, CH+, CH2, CH2+, CH3, CH3+, CH4, CH4+, CH5+, CO, CO+, HCO+, He, He+, E"
osu_masses = [
    12.011,
    12.011,
    13.019,
    13.019,
    14.027,
    14.027,
    15.035,
    15.035,
    16.043,
    16.043,
    17.054,
    28.010,
    28.010,
    0.001,
    1.008,
    1.008,
    2.016,
    2.059,
    18.015,
    18.015,
    3.024,
    19.023,
    29.018,
    15.999,
    15.999,
    31.998,
    31.998,
    17.007,
    17.007,
]
