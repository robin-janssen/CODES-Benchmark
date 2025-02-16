import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Define base directory (two levels up from the current script)
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(BASE_DIR))

from codes.utils.data_utils import create_dataset  # noqa: E402
from datasets._data_generation.odes import FUNCS  # Importing FUNCS from odes.py


def generate_initial_conditions(
    num: int, ndim: int, sampling: Dict[str, Any], seed: int = None
) -> np.ndarray:
    """
    Generates initial conditions using Sobol sampling.

    Parameters
    ----------
    num : int
        Number of initial conditions to generate.
    ndim : int
        Number of dimensions/species.
    sampling : Dict[str, Any]
        Sampling configuration containing 'space' and 'bounds'.
    seed : int, optional
        Seed for the Sobol sequence generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of initial conditions with shape (num, ndim).
    """
    space = sampling.get("space", "linear")
    bounds = sampling.get("bounds", [(0.0, 1.0)] * ndim)

    if len(bounds) != ndim:
        logging.error(
            f"Number of bounds ({len(bounds)}) does not match number of dimensions ({ndim})."
        )
        sys.exit(1)

    # Extract lower and upper bounds for each dimension
    lower_bounds, upper_bounds = zip(*bounds)

    if space not in ["linear", "log"]:
        logging.error(f"Unsupported sampling space: '{space}'. Use 'linear' or 'log'.")
        sys.exit(1)

    if space == "log":
        # Ensure all bounds are positive
        if any(lb <= 0 or ub <= 0 for lb, ub in bounds):
            logging.error("All bounds must be positive for log-space sampling.")
            sys.exit(1)
        # Take logarithm of bounds
        lower_bounds = np.log(lower_bounds)
        upper_bounds = np.log(upper_bounds)

    # Initialize Sobol sampler
    sampler = qmc.Sobol(d=ndim, scramble=True, seed=seed)
    try:
        sample = sampler.random_base2(m=int(np.ceil(np.log2(num))))
    except ValueError:
        # If num is not a power of 2, generate the next higher power
        m = int(np.ceil(np.log2(num)))
        sample = sampler.random_base2(m=m)
    # Truncate to 'num' samples
    sample = sample[:num]

    # Scale samples to the specified bounds
    sample = qmc.scale(sample, lower_bounds, upper_bounds)

    if space == "log":
        # Exponentiate to get back to original scale
        sample = np.exp(sample)

    return sample


def generate_trajectory(
    func: Callable[[float, np.ndarray], np.ndarray],
    timesteps: np.ndarray,
    initial_condition: np.ndarray,
) -> np.ndarray:
    """
    Generates a single trajectory for the given ODE system.

    Parameters
    ----------
    func : Callable
        The ODE function.
    timesteps : np.ndarray
        Array of time points.
    initial_condition : np.ndarray
        Initial condition for the trajectory.

    Returns
    -------
    np.ndarray
        Generated trajectory with shape (n_timesteps, dim).
    """
    sol = solve_ivp(
        func,
        [timesteps[0], timesteps[-1]],
        initial_condition,
        t_eval=timesteps,
        method="DOP853",
        atol=1e-8,
        rtol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol.y.T


def create_data(
    num: int,
    func: Callable[[float, np.ndarray], np.ndarray],
    timesteps: np.ndarray,
    dim: int,
    sampling: Dict[str, Any],
    seed: int = None,
) -> np.ndarray:
    """
    Generates multiple trajectories for a given ODE system using Sobol sampling.

    Parameters
    ----------
    num : int
        Number of trajectories to generate.
    func : Callable
        The ODE function.
    timesteps : np.ndarray
        Array of time points.
    dim : int
        Number of dimensions/species.
    sampling : Dict[str, Any]
        Sampling configuration containing 'space' and 'bounds'.
    seed : int, optional
        Seed for the Sobol sequence generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of generated trajectories with shape (num, n_timesteps, dim).
    """
    logging.info(f"Generating {num} initial conditions using Sobol sampling...")
    initial_conditions = generate_initial_conditions(num, dim, sampling, seed=seed)

    data = np.empty((num, len(timesteps), dim))

    # Use tqdm for a progress bar
    with tqdm(total=num, desc="Generating trajectories") as pbar:
        for i in range(num):
            try:
                data[i] = generate_trajectory(func, timesteps, initial_conditions[i])
            except RuntimeError as e:
                logging.error(f"Trajectory {i+1}/{num} generation failed: {e}")
                sys.exit(1)
            pbar.update(1)  # Update progress bar after each trajectory generation

    return data


def prepare_dataset_directory(name: str, force: bool = False) -> Path:
    """
    Ensures that the dataset directory does not exist to prevent conflicts.

    Parameters
    ----------
    name : str
        Name of the dataset.
    force : bool, optional
        If True, existing dataset will be overwritten without prompt.

    Returns
    -------
    Path
        The path to the dataset directory.
    """
    dataset_path = BASE_DIR / "datasets" / name
    if dataset_path.exists():
        if force:
            logging.info(f"Overwriting existing dataset directory: {dataset_path}")
            try:
                import shutil

                shutil.rmtree(dataset_path)
            except Exception as e:
                logging.error(f"Failed to remove existing dataset directory: {e}")
                sys.exit(1)
        else:
            response = input(
                f"The data directory '{dataset_path}' already exists. "
                "Press Enter to overwrite it or type 'cancel' to exit: "
            )
            if response.lower() == "cancel":
                logging.info("Operation cancelled by the user.")
                sys.exit(0)
            logging.info(f"Removing existing dataset directory: {dataset_path}")
            try:
                import shutil

                shutil.rmtree(dataset_path)
            except Exception as e:
                logging.error(f"Failed to remove existing dataset directory: {e}")
                sys.exit(1)
    return dataset_path


def main():
    args = parse_args()

    # Retrieve function parameters
    func_info = FUNCS.get(args.func)
    if not func_info:
        logging.error(f"Function '{args.func}' is not supported.")
        sys.exit(1)

    # Validate that labels are defined
    labels = func_info.get("labels")
    if not labels:
        logging.error(f"No labels defined for function '{args.func}'.")
        sys.exit(1)

    if len(labels) != func_info["ndim"]:
        logging.error(
            f"Number of labels ({len(labels)}) does not match number of species ({func_info['ndim']})."
        )
        sys.exit(1)

    # Prepare dataset directory
    prepare_dataset_directory(args.name, force=args.force)

    # Retrieve function parameters
    func = func_info["func"]
    timesteps = func_info["tsteps"]
    dim = func_info["ndim"]
    sampling = func_info.get(
        "sampling", {"space": "linear", "bounds": [(0.0, 1.0)] * dim}
    )

    # Generate datasets
    logging.info("Generating training data...")
    data_train = create_data(
        num=args.num_train,
        func=func,
        timesteps=timesteps,
        dim=dim,
        sampling=sampling,
        seed=args.seed,
    )

    logging.info("Generating test data...")
    data_test = create_data(
        num=args.num_test,
        func=func,
        timesteps=timesteps,
        dim=dim,
        sampling=sampling,
        seed=args.seed,
    )

    logging.info("Generating validation data...")
    data_val = create_data(
        num=args.num_val,
        func=func,
        timesteps=timesteps,
        dim=dim,
        sampling=sampling,
        seed=args.seed,
    )

    # Save datasets
    try:
        create_dataset(
            name=args.name,
            train_data=data_train,
            test_data=data_test,
            val_data=data_val,
            timesteps=timesteps,
            labels=labels,
        )
        logging.info(f"Dataset '{args.name}' created successfully.")
    except FileExistsError:
        logging.error(
            f"Dataset '{args.name}' already exists. Use the '--force' flag to overwrite."
        )
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while creating the dataset: {e}")
        sys.exit(1)


def parse_args() -> ArgumentParser:
    """
    Parses command-line arguments for the dataset generation script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments as a namespace.
    """
    parser = ArgumentParser(
        description="Generate datasets for ODE systems and save them in HDF5 format."
    )
    parser.add_argument(
        "--num_train",
        "-tr",
        type=int,
        default=700,
        help="Number of training trajectories to generate (default: 700).",
    )
    parser.add_argument(
        "--num_test",
        "-te",
        type=int,
        default=100,
        help="Number of test trajectories to generate (default: 100).",
    )
    parser.add_argument(
        "--num_val",
        "-va",
        type=int,
        default=200,
        help="Number of validation trajectories to generate (default: 200).",
    )
    parser.add_argument(
        "--func",
        "-f",
        type=str,
        choices=FUNCS.keys(),
        default="simple_ode",
        help=f"Name of the function to generate data for (default: 'lotka_volterra'). Choices: {list(FUNCS.keys())}.",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="simple_ode",
        help="Name of the dataset (default: 'lotka_volterra_new').",
    )
    parser.add_argument(
        "--force",
        "-fo",
        action="store_true",
        help="Overwrite existing dataset without prompting (default: False).",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Seed for random number generator (default: None).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
