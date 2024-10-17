# TODO: move this to an appropriate location

import os

# Add codes package to the path (two keys up)
import sys
from argparse import ArgumentParser
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(1, "../..")

from codes.utils.data_utils import create_dataset


def lotka_volterra(t, n):
    """
    Differential equations for a Lotka-Volterra system with three predators and three prey.

    Parameters
    ----------
    t : float
        Time
    n : array
        Array of concentrations of species p1, p2, p3, q1, q2, and q3.

    Returns
    -------
    array
        Array of the derivatives of the abundances of species p1, p2, p3, q1, q2, and q3.
    """
    p1, p2, p3, q1, q2, q3 = n[0], n[1], n[2], n[3], n[4], n[5]
    return np.array(
        [
            0.5 * p1 - 0.02 * p1 * q1 - 0.01 * p1 * q2,
            0.6 * p2 - 0.03 * p2 * q1 - 0.015 * p2 * q3,
            0.4 * p3 - 0.01 * p3 * q2 - 0.025 * p3 * q3,
            -0.1 * q1 + 0.005 * p1 * q1 + 0.007 * p2 * q1,
            -0.08 * q2 + 0.006 * p1 * q2 + 0.009 * p3 * q2,
            -0.12 * q3 + 0.008 * p2 * q3 + 0.01 * p3 * q3,
        ]
    )


def reaction(t, n):
    """
    Differential equations for a simple chemical reaction system.

    Parameters
    ----------
    t : float
        Time
    n : array
        Array of concentrations of species s1, s2, s3, s4, s5, and s6.

    Returns
    -------
    array
        Array of the derivatives of the abundances of species s1, s2, s3, s4, s5, and s6.
    """
    s1, s2, s3, s4, s5, _ = n[0], n[1], n[2], n[3], n[4], n[5]
    return np.array(
        [
            -0.1 * s1 + 0.1 * s2,
            0.1 * s1 - 0.15 * s2 + 0.05 * s3,
            0.15 * s2 - 0.1 * s3 + 0.03 * s4,
            0.1 * s3 - 0.07 * s4 + 0.01 * s5,
            0.07 * s4 - 0.05 * s5,
            0.05 * s5,
        ]
    )


FUNCS = {
    "lotka_volterra": {
        "func": lotka_volterra,
        "tsteps": np.linspace(0, 100, 100),
        "ndim": 6,
    },
    "reaction": {"func": reaction, "tsteps": np.linspace(0, 10, 100), "ndim": 6},
}


def create_data(num: int, func: Callable, timesteps: np.ndarray, dim: int):
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
    n_tsteps = timesteps.shape[0]
    data = np.empty((num, n_tsteps, dim))
    for i in range(num):
        n0 = np.random.rand(dim)
        sol = solve_ivp(func, [0, n_tsteps], n0, t_eval=timesteps)
        data[i] = sol.y.T
    return data, timesteps


def main(args):

    # Switch cwd to the root directory
    os.chdir("../..")
    if os.path.exists(f"datasets/{args.name}"):
        res = input(
            f"The data directory 'datasets/{args.name}' already exists. Press Enter to overwrite it."
        )
        if res != "":
            return
        os.system(f"rm -r datasets/{args.name}")

    if not FUNCS.get(args.func):
        print(f"Function {args.func} not found")
        return
    function = FUNCS[args.func]["func"]
    timesteps = FUNCS[args.func]["tsteps"]
    dim = FUNCS[args.func]["ndim"]

    print("generating training data...")
    data_train, timesteps = create_data(
        num=args.num_train,
        func=function,
        timesteps=timesteps,
        dim=dim,
    )
    print("generating test data...")
    data_test, _ = create_data(
        num=args.num_test,
        func=function,
        timesteps=timesteps,
        dim=dim,
    )
    print("generating validation data...")
    data_val, _ = create_data(
        num=args.num_val,
        func=function,
        timesteps=timesteps,
        dim=dim,
    )
    print("saving data...")
    create_dataset(
        name=args.name,
        train_data=data_train,
        test_data=data_test,
        val_data=data_val,
        timesteps=timesteps,
        labels=["A", "B", "C", "D", "E", "F"],
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
    parser.add_argument(
        "--func",
        default="lotka_volterra",
        type=str,
        help="Name of the function to generate data for",
    )
    parser.add_argument(
        "--name",
        default="simple_ode",
        type=str,
        help="Name of the dataset",
    )
    args = parser.parse_args()
    main(args)
