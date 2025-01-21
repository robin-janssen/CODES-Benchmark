import argparse
import concurrent.futures
import os
import queue
import threading
import time
from typing import Dict, List

import optuna
from optuna.pruners import HyperbandPruner, NopPruner
from optuna.trial import TrialState
from tqdm import tqdm

# Import your existing utilities from optuna_fcts.py and other modules
from codes.tune.optuna_fcts import load_yaml_config, training_run
from codes.utils import nice_print


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-architecture Optuna tuning with advanced parallelization."
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="lotkavolterra",
        help="Main study identifier. Separate sub-studies will be created for each architecture.",
    )
    return parser.parse_args()


class ParallelOptunaRunner:
    def __init__(self, config: Dict, main_study_name: str):
        """
        Initialize the ParallelOptunaRunner.

        Args:
            config (dict): The main YAML config with keys like "surrogates", "devices", etc.
            main_study_name (str): The name of the top-level study (e.g., 'lotkavolterra').
        """
        self.config = config
        self.main_study_name = main_study_name
        self.surrogates = config["surrogates"]

        # Initialize studies, sub_configs, arch_names, and trials needed
        self.studies: List[optuna.Study] = []
        self.n_new_trials_needed: List[int] = []
        self.sub_configs: List[Dict] = []
        self.arch_names: List[str] = []

        # Initialize the device queue with all available devices
        self.device_queue = queue.Queue()
        for dev in config["devices"]:
            self.device_queue.put(dev)

        # Initialize completed trials count
        self.completed_new: List[int] = []

        # Initialize locks for thread-safe operations
        self.schedule_lock = threading.Lock()

        # Initialize a ThreadPoolExecutor with max_workers equal to the number of devices
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(config["devices"])
        )

        # Initialize a set to keep track of in-flight futures
        self.futures_in_flight = set()

        # Initialize a list to hold callbacks for when trials are done
        self.callbacks = []

        # Initialize a progress bar for total new trials
        self.pbar = None

    def setup_studies(self):
        """
        Setup Optuna studies for each surrogate architecture.
        Determines how many new trials need to be run based on existing trials.
        """
        for surr in self.surrogates:
            arch_name = surr["name"]
            study_name = f"{self.main_study_name}_{arch_name.lower()}"

            # Build the DB URL
            db_url = self.config.get("storage_url", None)
            if db_url is None:
                # Use local SQLite
                db_path = os.path.join(
                    f"optuna_runs/{self.main_study_name}", f"{arch_name}.db"
                )
                db_url = f"sqlite:///{db_path}"

            # Setup sampler and pruner
            sampler = optuna.samplers.TPESampler(seed=self.config["seed"])

            if self.config.get("prune", True):
                epochs = surr["epochs"]  # Assuming 'epochs' is per surrogate
                pruner = HyperbandPruner(
                    min_resource=epochs // 8, max_resource=epochs, reduction_factor=2
                )
            else:
                pruner = NopPruner()

            # Create or load the study
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                storage=db_url,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )

            # Determine how many new trials are needed
            n_trials = surr.get("trials", self.config.get("trials", None))
            if n_trials is None:
                raise ValueError(
                    f"Number of trials not specified for surrogate '{arch_name}'."
                )

            already_done = sum(
                1
                for t in study.get_trials(deepcopy=False)
                if t.state
                in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING)
            )
            to_run = max(0, n_trials - already_done)

            self.studies.append(study)
            self.n_new_trials_needed.append(to_run)
            self.arch_names.append(arch_name)
            self.completed_new.append(0)

            # Build a sub-config for this architecture
            sub_config = {
                "batch_size": surr["batch_size"],
                "dataset": self.config["dataset"],
                "epochs": surr["epochs"],
                "n_trials": n_trials,  # total desired
                "seed": self.config["seed"],
                "surrogate": {"name": arch_name},
                "optuna_params": surr["optuna_params"],
                "prune": self.config.get("prune", True),
                "storage_url": self.config.get("storage_url", None),
                "optuna_logging": self.config.get("optuna_logging", False),
                "use_optimal_params": self.config.get("use_optimal_params", False),
                # "devices" will be assigned dynamically
            }
            self.sub_configs.append(sub_config)

    def initialize_progress_bar(self):
        """Initialize the tqdm progress bar based on total new trials."""
        total_new_trials = sum(self.n_new_trials_needed)
        if total_new_trials == 0:
            print(
                "All sub-studies already have the requested number of trials. Nothing to do."
            )
            return

        self.pbar = tqdm(
            total=total_new_trials, desc="Total new trials", position=0, leave=True
        )

    def run_one_trial(
        self, arch_index: int, trial: optuna.trial.FrozenTrial, device: str
    ):
        """
        Executes a single trial for a given architecture on a specified device.

        Args:
            arch_index (int): Index of the architecture/sub-study.
            trial (optuna.trial.FrozenTrial): The trial object.
            device (str): The device to run the trial on.
        """
        study = self.studies[arch_index]
        sub_cfg = self.sub_configs[arch_index]

        try:
            # Assign the device to the sub-config for this trial
            sub_cfg["devices"] = [device]

            # Run the training and obtain the loss
            loss = training_run(trial, device, sub_cfg, study.study_name)

            # Tell the study the result
            with self.schedule_lock:
                study.tell(trial, value=loss, state=TrialState.COMPLETE)

        except optuna.TrialPruned:
            # Handle pruned trials
            with self.schedule_lock:
                study.tell(trial, state=TrialState.PRUNED)

        except Exception as e:
            # Handle other exceptions and mark trial as failed
            with self.schedule_lock:
                study.tell(trial, state=TrialState.FAIL)
            print(
                f"Architecture '{self.arch_names[arch_index]}' trial failed with exception: {e}"
            )

        finally:
            # Release the device back to the queue
            self.device_queue.put(device)

    def trial_done_callback(self, future: concurrent.futures.Future, arch_index: int):
        """
        Callback function executed when a trial is completed.

        Args:
            future (concurrent.futures.Future): The future object.
            arch_index (int): Index of the architecture/sub-study.
        """
        exc = future.exception()
        with self.schedule_lock:
            if exc is None:
                # Trial completed successfully
                self.completed_new[arch_index] += 1
                if self.pbar:
                    self.pbar.update(1)
            elif isinstance(exc, optuna.TrialPruned):
                # Trial was pruned
                self.completed_new[arch_index] += 1
                if self.pbar:
                    self.pbar.update(1)
            else:
                # Trial failed with an unexpected exception
                print(
                    f"Architecture '{self.arch_names[arch_index]}' trial encountered an exception: {exc}"
                )

            # Remove the future from the in-flight set
            self.futures_in_flight.remove(future)

            # Attempt to schedule a new trial for this architecture if needed
            if self.completed_new[arch_index] < self.n_new_trials_needed[arch_index]:
                self.schedule_one_new_trial(arch_index)

    def schedule_one_new_trial(self, arch_index: int):
        """
        Attempts to schedule a single new trial for the specified architecture.

        Args:
            arch_index (int): Index of the architecture/sub-study.
        """
        # Check if more trials are needed for this architecture
        if self.completed_new[arch_index] >= self.n_new_trials_needed[arch_index]:
            return

        # Attempt to get a free device without blocking
        try:
            device = self.device_queue.get_nowait()
        except queue.Empty:
            # No devices are currently available
            return

        # Ask the study for a new trial
        study = self.studies[arch_index]
        try:
            trial = study.ask()
        except optuna.exceptions.OptunaError as e:
            print(
                f"Error asking for a new trial in architecture '{self.arch_names[arch_index]}': {e}"
            )
            # Release the device back to the queue
            self.device_queue.put(device)
            return

        # Submit the trial to the ThreadPoolExecutor
        future = self.pool.submit(self.run_one_trial, arch_index, trial, device)
        self.futures_in_flight.add(future)

        # Attach the callback to handle completion
        future.add_done_callback(
            lambda fut, idx=arch_index: self.trial_done_callback(fut, idx)
        )

    def initial_scheduling(self):
        """
        Performs the initial scheduling of trials across all architectures.
        """
        with self.schedule_lock:
            for arch_index in range(len(self.studies)):
                while (
                    self.completed_new[arch_index]
                    < self.n_new_trials_needed[arch_index]
                ):
                    # Attempt to schedule as many trials as possible per architecture
                    prev_size = len(self.futures_in_flight)
                    self.schedule_one_new_trial(arch_index)
                    new_size = len(self.futures_in_flight)
                    if new_size == prev_size:
                        # No new trial was scheduled for this architecture
                        break

    def wait_for_completion(self):
        """
        Waits until all required trials have been completed.
        """
        while True:
            with self.schedule_lock:
                if all(
                    self.completed_new[i] >= self.n_new_trials_needed[i]
                    for i in range(len(self.studies))
                ):
                    break
            time.sleep(0.3)  # Sleep briefly to avoid tight loop

        # Wait for all in-flight futures to complete
        concurrent.futures.wait(self.futures_in_flight)
        self.pool.shutdown(wait=True)

    def run(self):
        """
        Executes the parallel tuning process.
        """
        self.setup_studies()
        self.initialize_progress_bar()

        # If there are no new trials to run, exit early
        if sum(self.n_new_trials_needed) == 0:
            print(
                "All sub-studies already have the requested number of trials. Nothing to do."
            )
            if self.pbar:
                self.pbar.close()
            return

        # Perform initial scheduling
        self.initial_scheduling()

        # Wait until all trials are completed
        self.wait_for_completion()

        # Close the progress bar
        if self.pbar:
            self.pbar.close()

        # Print final summaries
        print("\nFinal Results:")
        for i, study in enumerate(self.studies):
            arch_name = self.arch_names[i]
            if study.best_trial is not None:
                print(
                    f"Architecture: {arch_name}, Best Value: {study.best_trial.value:.6f}"
                )
                print(f"  Best Trial Params: {study.best_trial.params}")
            else:
                print(f"Architecture: {arch_name}, No trials completed.")
        print("All sub-studies complete!")


def run_multi_arch_studies_with_dynamic_device_allocation(
    config: dict, main_study_name: str
):
    """
    Orchestrates the parallel execution of multiple Optuna studies across available devices.

    Args:
        config (dict): The main YAML config with keys like "surrogates", "devices", etc.
        main_study_name (str): The name of the top-level study (e.g., 'lotkavolterra').
    """
    runner = ParallelOptunaRunner(config, main_study_name)
    runner.run()


def main():
    """Main entry point for the advanced parallel tuning script."""
    nice_print("Starting advanced Optuna tuning (ask-and-tell)")
    args = parse_arguments()
    config_path = os.path.join("optuna_runs", args.study_name, "optuna_config.yaml")
    config = load_yaml_config(config_path)

    optuna_logging = config.get("optuna_logging", False)
    if not optuna_logging:
        print("Optuna logging disabled. No intermediate results will be printed.")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Check if 'surrogates' are defined in the config
    if "surrogates" not in config:
        print("No 'surrogates' defined in config. Exiting.")
        return

    run_multi_arch_studies_with_dynamic_device_allocation(config, args.study_name)

    nice_print("Optuna tuning completed!")


if __name__ == "__main__":
    main()
