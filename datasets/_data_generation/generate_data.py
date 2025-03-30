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
from datasets._data_generation.odes import FUNCS  # Import FUNCS from odes.py


def generate_initial_conditions(
    num: int, ndim: int, sampling: Dict[str, Any], seed: int = None
) -> np.ndarray:
    """Generates initial conditions using standard Sobol sampling."""
    space = sampling.get("space", "linear")
    bounds = sampling.get("bounds", [(0.0, 1.0)] * ndim)
    if len(bounds) != ndim:
        logging.error(
            f"Number of bounds ({len(bounds)}) does not match dimensions ({ndim})."
        )
        sys.exit(1)
    lower_bounds, upper_bounds = zip(*bounds)
    if space not in ["linear", "log"]:
        logging.error(f"Unsupported sampling space: '{space}'.")
        sys.exit(1)
    if space == "log":
        if any(lb <= 0 or ub <= 0 for lb, ub in bounds):
            logging.error("All bounds must be positive for log-space sampling.")
            sys.exit(1)
        lower_bounds = np.log(lower_bounds)
        upper_bounds = np.log(upper_bounds)
    sampler = qmc.Sobol(d=ndim, scramble=True, seed=seed)
    sample = sampler.random_base2(m=int(np.ceil(np.log2(num))))
    sample = sample[:num]
    sample = qmc.scale(sample, lower_bounds, upper_bounds)
    if space == "log":
        sample = np.exp(sample)
    return sample


def generate_trajectory(
    func: Callable[[float, np.ndarray], np.ndarray],
    timesteps: np.ndarray,
    initial_condition: np.ndarray,
    solver_options: Dict[str, Any] = None,
    log_time: bool = False,
    final_transform: bool = False,
) -> np.ndarray:
    """
    Generates a single trajectory for the given ODE system.

    If log_time is True, the integration is performed in the transformed (log₁₀) time.
    At the end, if final_transform is True, the state is exponentiated (base 10)
    to return the solution in linear space.
    """
    if solver_options is None:
        solver_options = {"method": "DOP853", "atol": 1e-8, "rtol": 1e-8}
    if log_time:
        # Set tspan and t_eval in log–space.
        spy = 365.0 * 24.0 * 3600.0
        tmin = np.log10(1e-6 * spy)
        t_end = np.log10(timesteps[-1])
        tspan = (0, t_end)
        t_eval = np.linspace(tmin, t_end, len(timesteps))
        sol = solve_ivp(func, tspan, initial_condition, t_eval=t_eval, **solver_options)
    else:
        sol = solve_ivp(
            func,
            [timesteps[0], timesteps[-1]],
            initial_condition,
            t_eval=timesteps,
            **solver_options,
        )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    sol_data = sol.y.T  # shape (n_timesteps, ndim)
    if final_transform:
        # Since the state was integrated in log₁₀-space, return 10^(state)
        sol_data = np.power(10, sol_data)
    return sol_data


def create_data(
    num: int,
    func: Callable[[float, np.ndarray], np.ndarray],
    timesteps: np.ndarray,
    dim: int,
    sampling: Dict[str, Any],
    seed: int = None,
    init_func: Callable[[int], np.ndarray] = None,
    solver_options: Dict[str, Any] = None,
    log_time: bool = False,
    final_transform: bool = False,
) -> np.ndarray:
    """
    Generates multiple trajectories for a given ODE system.

    If an initial-condition function (init_func) is provided, it is used to generate
    the initial conditions; otherwise, standard Sobol sampling is used.
    """
    if init_func is not None:
        logging.info("Generating initial conditions using custom procedure.")
        initial_conditions = init_func(num)
    else:
        logging.info(f"Generating {num} initial conditions using Sobol sampling...")
        initial_conditions = generate_initial_conditions(num, dim, sampling, seed=seed)
    data = np.empty((num, len(timesteps), dim))
    with tqdm(total=num, desc="Generating trajectories") as pbar:
        for i in range(num):
            data[i] = generate_trajectory(
                func,
                timesteps,
                initial_conditions[i],
                solver_options=solver_options,
                log_time=log_time,
                final_transform=final_transform,
            )
            pbar.update(1)
    return data


def prepare_dataset_directory(name: str, force: bool = False) -> Path:
    """
    Ensures that the dataset directory does not already exist.
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
                sys.exit(1)
            logging.info(f"Removing existing dataset directory: {dataset_path}")
            try:
                import shutil

                shutil.rmtree(dataset_path)
            except Exception as e:
                logging.error(f"Failed to remove existing dataset directory: {e}")
                sys.exit(1)
    return dataset_path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Generate datasets for ODE systems and save them in HDF5 format."
    )
    parser.add_argument("--num_train", "-tr", type=int, default=700)
    parser.add_argument("--num_test", "-te", type=int, default=100)
    parser.add_argument("--num_val", "-va", type=int, default=200)
    parser.add_argument(
        "--func",
        "-f",
        type=str,
        choices=FUNCS.keys(),
        default="osu",
        help=f"Name of the function to generate data for. Choices: {list(FUNCS.keys())}.",
    )
    parser.add_argument("--name", "-n", type=str, default="osutest")
    parser.add_argument("--force", "-fo", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    func_info = FUNCS.get(args.func)
    if not func_info:
        logging.error(f"Function '{args.func}' is not supported.")
        sys.exit(1)
    labels = func_info.get("labels")
    if not labels:
        logging.error(f"No labels defined for function '{args.func}'.")
        sys.exit(1)
    if len(labels) != func_info["ndim"]:
        logging.error(
            f"Number of labels ({len(labels)}) does not match number of species ({func_info['ndim']})."
        )
        sys.exit(1)

    prepare_dataset_directory(args.name, force=args.force)

    func = func_info["func"]
    timesteps = func_info["tsteps"]
    dim = func_info["ndim"]
    sampling = func_info.get(
        "sampling", {"space": "linear", "bounds": [(0.0, 1.0)] * dim}
    )
    init_func = func_info.get("init_func", None)
    solver_options = func_info.get(
        "solver_options", {"method": "DOP853", "atol": 1e-8, "rtol": 1e-8}
    )
    log_time = func_info.get("log_time", False)
    final_transform = func_info.get("final_transform", False)

    logging.info("Generating training data...")
    data_train = create_data(
        num=args.num_train,
        func=func,
        timesteps=timesteps,
        dim=dim,
        sampling=sampling,
        seed=args.seed,
        init_func=init_func,
        solver_options=solver_options,
        log_time=log_time,
        final_transform=final_transform,
    )
    logging.info("Generating test data...")
    data_test = create_data(
        num=args.num_test,
        func=func,
        timesteps=timesteps,
        dim=dim,
        sampling=sampling,
        seed=args.seed,
        init_func=init_func,
        solver_options=solver_options,
        log_time=log_time,
        final_transform=final_transform,
    )
    logging.info("Generating validation data...")
    data_val = create_data(
        num=args.num_val,
        func=func,
        timesteps=timesteps,
        dim=dim,
        sampling=sampling,
        seed=args.seed,
        init_func=init_func,
        solver_options=solver_options,
        log_time=log_time,
        final_transform=final_transform,
    )

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
    except Exception as e:
        logging.error(f"An error occurred while creating the dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
