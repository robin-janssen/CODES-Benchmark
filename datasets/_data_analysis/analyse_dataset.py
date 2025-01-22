import os
import sys
from argparse import ArgumentParser

import numpy as np

# Navigate up one level from the current working directory to access 'codes'
current_path = os.path.dirname(os.path.abspath(__file__))  # Path to this script
project_root = os.path.abspath(os.path.join(current_path, "../../"))  # Go up two levels

# Add the 'codes' directory to the system path
codes_path = os.path.join(project_root, "codes")
sys.path.insert(1, codes_path)
print(sys.path)

from codes import check_and_load_data, download_data
from datasets._data_analysis.data_plots import (  # plot_example_trajectories_poster,; plot_average_gradients_over_time,
    debug_numerical_errors_plot,
    plot_all_gradients_over_time,
    plot_all_trajectories_over_time,
    plot_example_trajectories,
    plot_initial_conditions_distribution,
)


def main(args):
    """
    Main function to analyse the dataset. It checks the dataset and loads the data.
    """
    log = False  # args.log
    debug = False  # args.debug
    qpp = 5
    # Load full data
    download_data(args.dataset)
    (
        full_train_data,
        full_test_data,
        full_val_data,
        timesteps,
        _,
        _,
        labels,
    ) = check_and_load_data(
        args.dataset,
        verbose=False,
        log=log,
        normalisation_mode="disable",
        tolerance=None,
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
        quantities_per_plot=qpp,
    )

    # plot_example_trajectories_poster(
    #     args.dataset,
    #     full_train_data,
    #     timesteps,
    #     save=True,
    #     labels=labels,
    #     sample_idx=7,
    # )

    full_data = np.concatenate([full_train_data, full_test_data, full_val_data], axis=0)

    if debug:
        debug_numerical_errors_plot(
            args.dataset,
            full_data,
            labels,
            max_quantities=10,
            threshold=1.2,
            max_faulty=5,
            quantities_per_plot=qpp,
        )

    # Plot initial conditions distribution
    plot_initial_conditions_distribution(
        args.dataset,
        full_train_data,
        full_val_data,
        chemical_names=labels,
        max_chemicals=50,
        log=log,
        quantities_per_plot=qpp,
    )

    # plot_average_gradients_over_time(
    #     args.dataset,
    #     full_data,
    #     labels,
    #     max_quantities=30,
    # )

    plot_all_trajectories_over_time(
        args.dataset,
        full_data,
        labels,
        max_quantities=30,
        quantities_per_plot=qpp,
    )

    plot_all_gradients_over_time(
        args.dataset,
        full_data,
        labels,
        max_quantities=30,
        quantities_per_plot=qpp,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="coupled_oscillators",
        type=str,
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to enable debugging plots.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Flag to enable logging.",
    )
    # args = parser.parse_args("--dataset simple_oscillator --debug --log".split())
    args = parser.parse_args()
    main(args)
