import os
import sys

import numpy as np

# Move two levels up from the current working directory if the script is run from _data_analysis
if os.path.basename(os.getcwd()) == "_data_analysis":
    current_path = os.path.dirname(os.path.abspath(__file__))  # Path to this script
    print("Current path:", current_path)
    project_root = os.path.abspath(
        os.path.join(current_path, "../../")
    )  # Go up two levels
    print("Project root:", project_root)
else:
    project_root = os.getcwd()  # Current working directory
    print("Current working directory:", project_root)

sys.path.insert(1, project_root)
print(sys.path)

from codes import check_and_load_data, download_data
from datasets._data_analysis.data_plots import (  # plot_example_trajectories_poster,; plot_average_gradients_over_time,
    debug_numerical_errors_plot,
    plot_all_trajectories_and_gradients,
    plot_example_trajectories,
    plot_initial_conditions_distribution,
)
from datasets._data_analysis.dataset_dict import dataset_dict


def main():
    """
    Main function to analyse the dataset. It checks the dataset and loads the data.
    """
    datasets = [
        "primordial_parametric",
        "primordial",
        "cloud_parametric",
        "cloud",
        "lv_parametric",
        "lv_parametric_no_params",
        "simple_reaction",
        "osutest2",
        "coupled_oscillators",
        "lotka_volterra",
        "simple_ode",
        # "osutest",
        # "osu2008",
        # "simple_primordial",
        # "branca_large_kyr",
        # "branca_large_myr",
        # "branca24",
        # "branca_norad",
    ]
    debug = True
    TITLE = True
    # Load full data
    for dataset in datasets:
        log = dataset_dict[dataset]["log"]
        qpp = dataset_dict[dataset]["qpp"]
        tolerance = dataset_dict[dataset]["tol"]
        download_data(dataset)
        (
            (full_train_data, full_test_data, full_val_data),
            _,
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

        num_quantities = full_data.shape[2]

        plot_all_trajectories_and_gradients(
            dataset,
            full_data,
            labels,
            max_quantities=num_quantities,
            quantities_per_plot=qpp,
            max_trajectories=1000,
            timesteps=timesteps,
            log=log,
            log_time=dataset_dict[dataset].get("log_time", False),
            show_title=TITLE,
        )

        plot_example_trajectories(
            dataset,
            full_data,
            timesteps,
            num_quantities=num_quantities,
            save=True,
            labels=labels,
            sample_idx=86,
            log=log,
            log_time=dataset_dict[dataset].get("log_time", False),
            quantities_per_plot=qpp,
            show_title=TITLE,
        )

        # Plot initial conditions distribution
        plot_initial_conditions_distribution(
            dataset,
            full_train_data,
            full_val_data,
            quantity_names=labels,
            max_quantities=453,
            log=log,
            quantities_per_plot=qpp,
            show_title=TITLE,
        )

        if debug:
            debug_numerical_errors_plot(
                dataset,
                full_data,
                labels,
                max_quantities=10,
                threshold=4,
                max_faulty=5,
                quantities_per_plot=qpp,
            )


if __name__ == "__main__":
    main()
