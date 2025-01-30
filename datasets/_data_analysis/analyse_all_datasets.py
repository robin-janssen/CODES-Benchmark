import os
import sys

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
    plot_all_trajectories_and_gradients,
    plot_example_trajectories,
    plot_initial_conditions_distribution,
)


def main():
    """
    Main function to analyse the dataset. It checks the dataset and loads the data.
    """
    datasets = [
        "coupled_oscillators",
        "simple_reaction",
        "osu2008",
        "simple_ode",
        "simple_primordial",
        "lotka_volterra",
        "branca_large_kyr",
        "branca_large_myr",
        "branca24",
        "branca_norad",
    ]
    logs = [False, True, True, True, True, True, True, True, True, True]
    qpps = [5, 3, 6, 3, 4, 3, 10, 10, 5, 5]
    tols = [None, 1e-30, 1e-30, 1e-30, 1e-30, 1e-30, 1e-30, 1e-30, 1e-30, 1e-30]
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

        full_data = np.concatenate(
            [full_train_data, full_test_data, full_val_data], axis=0
        )

        num_chems = full_data.shape[2]

        plot_all_trajectories_and_gradients(
            dataset,
            full_data,
            labels,
            max_quantities=num_chems,
            quantities_per_plot=qpp,
            max_trajectories=1000,
        )

        plot_example_trajectories(
            dataset,
            full_train_data,
            timesteps,
            num_chemicals=num_chems,
            save=True,
            labels=labels,
            sample_idx=7,
            log=log,
            quantities_per_plot=qpp,
        )

        # Plot initial conditions distribution
        plot_initial_conditions_distribution(
            dataset,
            full_train_data,
            full_val_data,
            chemical_names=labels,
            max_quantities=453,
            log=log,
            quantities_per_plot=qpp,
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


if __name__ == "__main__":
    main()
