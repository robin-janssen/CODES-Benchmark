import sys
from argparse import ArgumentParser

sys.path.insert(1, "../..")

from codes import check_and_load_data

from datasets.data_analysis.data_plots import (
    plot_example_trajectories,
    # plot_example_trajectories_paper,
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
        "--dataset", default="branca24", type=str, help="Name of the dataset."
    )
    args = parser.parse_args()
    main(args)
