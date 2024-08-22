# TODO: move this to an appropriate location

import os
from argparse import ArgumentParser

import numpy as np
from scipy.integrate import solve_ivp

from data.data_utils import create_dataset


def func(t, n):
    """
    Differential equations for a simple ODE system.

    Parameters
    ----------
    t : float
        Time
    n : array
        Array of concentrations of species A, B, C, D, and E.

    Returns
    -------
    array
        Array of the derivatives of the concentrations of species A, B, C, D, and E.
    """
    k = np.array([0.8, 0.5, 0.2])
    return np.array(
        [
            -k[0] * n[0] - k[2] * n[0] * n[2] / (t + 1),
            k[0] * n[0] - k[1] * n[1] + 2 * k[2] * n[0] * n[2],
            k[1] * n[1] - k[2] * n[0] * n[2],
            k[2] * n[0] * t,
            k[0] * n[0] / k[1] * n[2] - k[1] * n[0] * n[2],
        ]
    )


def create_data(num: int):
    """
    Create data for a simple ODE system.

    Parameters
    ----------
    num : int
        Number of trajectories to generate.

    Returns
    -------
    array
        Array of generated trajectories.
    """
    data = np.empty((num, 100, 5))
    timesteps = np.linspace(0, 10, 100)
    for i in range(num):
        n0 = np.random.rand(5)
        sol = solve_ivp(func, [0, 10], n0, t_eval=timesteps)
        data[i] = sol.y.T
    return data, timesteps


def main(args):

    if os.path.exists("data/simple_ode"):
        res = input(
            "The data directory 'data/simple_ode' already exists. Press Enter to overwrite it."
        )
        if res != "":
            return
        os.system("rm -r data/simple_ode")

    print("generating training data...")
    data_train, timesteps = create_data(args.num_train)
    print("generating test data...")
    data_test, _ = create_data(args.num_test)
    print("generating validation data...")
    data_val, _ = create_data(args.num_val)
    print("saving data...")
    create_dataset(
        "simple_ode",
        data_train,
        data_test,
        data_val,
        timesteps=timesteps,
        labels=["A", "B", "C", "D", "E"],
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--num_train",
        default=500,
        type=int,
        help="Number of training trajectories to generate.",
    )
    parser.add_argument(
        "--num_test",
        default=50,
        type=int,
        help="Number of test trajectories to generate.",
    )
    parser.add_argument(
        "--num_val",
        default=150,
        type=int,
        help="Number of validation trajectories to generate.",
    )
    args = parser.parse_args()
    main(args)
