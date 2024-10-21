import os
import sys
from argparse import ArgumentParser

# Navigate up one level from the current working directory to access 'codes'
current_path = os.path.dirname(os.path.abspath(__file__))  # Path to this script
project_root = os.path.abspath(os.path.join(current_path, "../../"))  # Go up two levels

# Add the 'codes' directory to the system path
codes_path = os.path.join(project_root, "codes")
print(f"Adding codes path: {codes_path}")
sys.path.insert(1, codes_path)
print("Current sys.path:", sys.path)

# Check if the codes directory is accessible
print("Contents of 'codes' directory:", os.listdir(codes_path))

# codes_path = os.path.join(os.getcwd(), "codes")
# print(codes_path)
# sys.path.insert(1, codes_path)
# print(sys.path)

from codes import check_and_load_data
from datasets.data_analysis.data_plots import (
    plot_example_trajectories,
    plot_initial_conditions_distribution,
)


def main(args):
    """
    Main function to analyse the dataset. It checks the dataset and loads the data.
    """
    log = True
    # Load full data
    (
        full_train_data,
        full_test_data,
        full_val_data,
        timesteps,
        _,
        data_params,
        labels,
    ) = check_and_load_data(
        args.dataset,
        verbose=False,
        log=log,
        normalisation_mode="disable",
    )

    # Plot example trajectories
    plot_example_trajectories(
        args.dataset,
        full_train_data,
        timesteps,
        num_chemicals=20,
        save=True,
        labels=labels,
        sample_idx=7,
        log=log,
    )

    # Plot initial conditions distribution
    plot_initial_conditions_distribution(
        args.dataset, full_train_data, chemical_names=labels, num_chemicals=50
    )

    # plot_example_trajectories_paper(
    #     args.dataset,
    #     full_train_data,
    #     timesteps,
    #     save=True,
    #     labels=labels,
    # )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", default="simple_ode", type=str, help="Name of the dataset."
    )
    args = parser.parse_args()
    main(args)
