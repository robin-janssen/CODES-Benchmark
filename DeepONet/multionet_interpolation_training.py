# This script is used to train a MultiONet model on the osu dataset to evaluate interpolation (and extrapolation) performance.
# To do so, the subset of data used for trainig will be increasingly sparse -
# either by leaving out some of the intermediate timesteps or by cutting off the end of the dataset.

import numpy as np

from DeepONet.dataloader import (
    create_dataloader_chemicals,
)

from DeepONet.config_classes import OChemicalTrainConfig
from DeepONet.train_multionet import (
    train_multionet_chemical,
    test_deeponet,
    load_multionet,
)

from train_utils import save_model

from plotting import (
    plot_chemical_examples,
    plot_chemicals_comparative,
    plot_relative_errors_over_time,
    plot_chemical_results,
    plot_losses,
)


def run(args):

    config = OChemicalTrainConfig()
    # config.device = args.device
    TRAIN = True
    args.vis = False
    config.device = args.device
    intervals = (9, 10)
    # cutoff = 50

    # Load the data
    # data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_1.npy")
    # data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_2.npy")
    # print(f"Loaded Branca data with shape: {data.shape}")
    # analyze_branca_data(data)
    # train_data, test_data, timesteps = prepare_branca_data(
    #     data, train_cut=500000, test_cut=100000
    # )
    full_train_data = np.load("data/osu_data/train_data.npy")
    config.train_size = full_train_data.shape[0]
    full_test_data = np.load("data/osu_data/test_data.npy")
    config.test_size = full_test_data.shape[0]
    print(f"Loaded Osu data with shape: {full_train_data.shape}/{full_test_data.shape}")
    full_timesteps = np.linspace(0, 99, 100)

    for interval in intervals:

        # Modify the data for interpolation testing
        train_data = full_train_data[:, ::interval]
        test_data = full_test_data[:, ::interval]

        # # Modify the data for extrapolation testing
        # train_data = train_data[:, :cutoff]
        # test_data = test_data[:, :cutoff]

        timesteps = full_timesteps[::interval]
        # timesteps = full_timesteps[:cutoff]
        config.N_timesteps = len(timesteps)

        if args.vis:
            plot_chemical_examples(
                train_data,
                num_chemicals=10,
                save=True,
                title="Chemical Examples (Osu Data)",
            )
            plot_chemicals_comparative(
                train_data,
                num_chemicals=10,
                save=True,
                title="Chemical Comparison (Osu Data)",
            )

        dataloader_train = create_dataloader_chemicals(
            train_data,
            timesteps,
            fraction=1,
            batch_size=config.batch_size,
            shuffle=True,
        )

        dataloader_test = create_dataloader_chemicals(
            test_data,
            timesteps,
            fraction=1,
            batch_size=config.batch_size,
            shuffle=False,
        )

        if TRAIN:
            multionet, train_loss, test_loss = train_multionet_chemical(
                config, dataloader_train, dataloader_test
            )

            # Save the MulitONet
            save_model(
                multionet,
                f"multionet_ochemicals_interp_{interval}",
                # f"multionet_ochemicals_extrap_{cutoff}",
                config,
                train_loss=train_loss,
                test_loss=test_loss,
                training_duration=train_multionet_chemical.duration,
            )

        else:
            model_path = f"models/04-29/multionet_ochemicals_interp_{interval}"
            # model_path = f"models/04-29/multionet_ochemicals_extrap_{cutoff}"
            multionet, train_loss, test_loss = load_multionet(
                config, config.device, model_path
            )

        average_error, predictions, ground_truth = test_deeponet(
            multionet,
            dataloader_test,
            N_timesteps=config.N_timesteps,
            reshape=True,
        )

        print(f"Average prediction error: {average_error:.3E}")

        if args.vis:

            plot_losses(
                (train_loss, test_loss),
                ("Train loss", "Test loss"),
                f"Losses (MultiONet on Osu Data, Interpolation interval: {interval})",
                # f"Losses (MultiONet on Osu Data, Extrapolation cutoff: {cutoff})",
                save=True,
            )

            errors = np.abs(predictions - ground_truth)
            relative_errors = errors / np.abs(ground_truth)

            plot_relative_errors_over_time(
                relative_errors,
                f"Relative errors over time (MultiONet on Osu Data, Interpolation interval: {interval})",
                # f"Relative errors over time (MultiONet on Osu Data, Extrapolation cutoff: {cutoff})",
                save=True,
            )

            plot_chemical_results(
                predictions=predictions,
                ground_truth=ground_truth,
                # names=extracted_chemicals,
                num_chemicals=10,
                title=f"Predictions of MultiONet on Osu Data (Interpolation interval: {interval})",
                # title=f"Predictions of MultiONet on Osu Data (Extrapolation cutoff: {cutoff})",
                save=True,
            )

    print("Done!")
