import os
import sys
from argparse import ArgumentParser

# Navigate up one level from the current working directory to access 'codes'
current_path = os.path.dirname(os.path.abspath(__file__))  # Path to this script
project_root = os.path.abspath(os.path.join(current_path, "../../"))  # Go up two levels

# Add the 'codes' directory to the system path
codes_path = os.path.join(project_root, "codes")
sys.path.insert(1, codes_path)
print(sys.path)

from codes import check_and_load_data, download_data
from datasets.data_analysis.data_plots import (  # plot_example_trajectories_poster,
    plot_all_gradients_over_time,
    plot_all_trajectories_over_time,
    plot_average_gradients_over_time,
    plot_example_trajectories,
    plot_initial_conditions_distribution,
)


def main(args):
    """
    Main function to analyse the dataset. It checks the dataset and loads the data.
    """
    log = True
    # Load full data
    download_data(args.dataset)
    (
        full_train_data,
        _,
        full_test_data,
        timesteps,
        _,
        _,
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
        num_chemicals=30,
        save=True,
        labels=labels,
        sample_idx=7,
        log=log,
    )

    # plot_example_trajectories_poster(
    #     args.dataset,
    #     full_train_data,
    #     timesteps,
    #     save=True,
    #     labels=labels,
    #     sample_idx=7,
    # )

    # Plot initial conditions distribution
    plot_initial_conditions_distribution(
        args.dataset,
        full_train_data,
        full_test_data,
        chemical_names=labels,
        max_chemicals=50,
        log=log,
    )

    plot_average_gradients_over_time(
        args.dataset,
        full_train_data,
        labels,
        max_quantities=30,
    )

    plot_all_gradients_over_time(
        args.dataset,
        full_train_data,
        labels,
        max_quantities=30,
    )

    plot_all_trajectories_over_time(
        args.dataset,
        full_train_data,
        labels,
        max_quantities=30,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", default="simple_reaction", type=str, help="Name of the dataset."
    )
    args = parser.parse_args()
    main(args)
