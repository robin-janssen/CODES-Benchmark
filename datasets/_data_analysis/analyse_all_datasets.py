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
    # datasets = [
    #     "coupled_oscillators",
    #     "simple_reaction",
    #     "osu2008",
    #     "simple_ode",
    #     "simple_primordial",
    #     "lotka_volterra",
    #     "branca_large_kyr",
    #     "branca_large_myr",
    #     "branca24",
    #     "branca_norad",
    # ]
    # logs = [False, True, True, True, True, True, True, True, True, True]
    # qpps = [5, 3, 6, 3, 4, 3, 10, 10, 5, 5]
    # tols = [None, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20]
    # debug = True
    datasets = [
        "branca_large_kyr",
        "branca_large_myr",
    ]
    logs = [True, True]
    qpps = [15, 15]
    tols = [1e-30, 1e-30]
    debug = False
    # Load full data
    for dataset, log, qpp, tolerance in zip(datasets, logs, qpps, tols):
        download_data(dataset)
        (
            full_train_data,
            full_test_data,
            full_val_data,
            timesteps,
            _,
            _,
            labels,
        ) = check_and_load_data(
            dataset,
            verbose=False,
            log=log,
            normalisation_mode="disable",
            tolerance=tolerance,
        )

        # Plot example trajectories
        plot_example_trajectories(
            dataset,
            full_train_data,
            timesteps,
            num_chemicals=75,
            save=True,
            labels=labels,
            sample_idx=7,
            log=log,
            quantities_per_plot=qpp,
        )

        # plot_example_trajectories_poster(
        #     dataset,
        #     full_train_data,
        #     timesteps,
        #     save=True,
        #     labels=labels,
        #     sample_idx=7,
        # )

        full_data = np.concatenate(
            [full_train_data, full_test_data, full_val_data], axis=0
        )

        if debug:
            debug_numerical_errors_plot(
                dataset,
                full_data,
                labels,
                max_quantities=10,
                threshold=1.2,
                max_faulty=5,
                quantities_per_plot=qpp,
            )

        # Plot initial conditions distribution
        plot_initial_conditions_distribution(
            dataset,
            full_train_data,
            full_val_data,
            chemical_names=labels,
            max_chemicals=75,
            log=log,
            quantities_per_plot=qpp,
        )

        # plot_average_gradients_over_time(
        #     dataset,
        #     full_data,
        #     labels,
        #     max_quantities=30,
        # )

        if full_data.shape[2] < 51:
            plot_all_trajectories_over_time(
                dataset,
                full_data,
                labels,
                max_quantities=50,
                quantities_per_plot=qpp,
            )

            plot_all_gradients_over_time(
                dataset,
                full_data,
                labels,
                max_quantities=50,
                quantities_per_plot=qpp,
            )
        else:


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="osu2008",
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
