# run_tuning.py

import argparse
import os
import queue
import subprocess
import sys
import time

import optuna
import psycopg2
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from psycopg2 import sql
from tqdm import tqdm

from codes.tune import create_objective, load_yaml_config
from codes.utils import nice_print


def check_postgres_running(config):
    """
    Check if PostgreSQL server is running.

    Args:
        config (dict): Configuration dictionary.

    Raises:
        Exception: If PostgreSQL server is not running.
    """
    db_config = config.get("postgres_config", {})
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "optuna_user")
    password = db_config.get("password", "")

    try:
        # Attempt to connect to the 'postgres' database
        conn = psycopg2.connect(
            dbname="postgres",
            user=user,
            password=password,
            host=host,
            port=port,
            connect_timeout=5,
        )
        conn.close()
    except psycopg2.OperationalError:
        raise Exception("PostgreSQL server is not running.")


def start_postgres_server(config):
    """
    Start the PostgreSQL server.

    Args:
        config (dict): Configuration dictionary.

    Raises:
        Exception: If the server fails to start.
    """
    db_config = config.get("postgres_config", {})
    data_dir = db_config.get("data_dir", os.path.expanduser("~/postgres/data"))
    log_file = db_config.get("log_file", os.path.expanduser("~/postgres/logfile"))
    database_folder = db_config.get("database_folder", os.path.expanduser("~/postgres"))

    # Path to the pg_ctl binary
    pg_ctl_path = os.path.join(database_folder, "bin", "pg_ctl")

    # Check if data_dir exists
    if not os.path.exists(data_dir):
        raise Exception(f"PostgreSQL data directory '{data_dir}' does not exist.")

    # Check if pg_ctl exists
    if not os.path.exists(pg_ctl_path):
        raise Exception(
            f"pg_ctl not found at '{pg_ctl_path}'. Please check your database_folder configuration."
        )

    try:
        print("Starting PostgreSQL server...")
        subprocess.run(
            [pg_ctl_path, "-D", data_dir, "-l", log_file, "start"], check=True
        )
        print("PostgreSQL server started successfully.")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to start PostgreSQL server: {e}")


def initialize_postgres(config, study_folder_name):
    """
    Initialize PostgreSQL database for Optuna studies.

    Args:
        config (dict): Configuration dictionary.
        study_folder_name (str): Name of the study folder to use as default database name.

    Returns:
        str: PostgreSQL connection URL.
    """
    db_config = config.get("postgres_config", {})
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "optuna_user")
    password = db_config.get("password", "")
    db_name = db_config.get("db_name", study_folder_name)

    try:
        # Connect to default 'postgres' database
        conn = psycopg2.connect(
            dbname="postgres",
            user=user,
            password=password,
            host=host,
            port=port,
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if the database exists
        cursor.execute(
            sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [db_name]
        )
        if cursor.fetchone():
            print(f"Database '{db_name}' already exists.")

            # Ask user whether to use or overwrite the database
            while True:
                user_choice = (
                    input(
                        f"Database '{db_name}' exists. Do you want to (U)se it or (O)verwrite it? [U/O]: "
                    )
                    .strip()
                    .lower()
                )
                if user_choice == "u":
                    print(f"Using existing database '{db_name}'.")
                    break
                elif user_choice == "o":
                    print(f"Overwriting database '{db_name}'.")
                    # Terminate existing connections to the database
                    cursor.execute(
                        sql.SQL(
                            """
                            SELECT pg_terminate_backend(pg_stat_activity.pid)
                            FROM pg_stat_activity
                            WHERE pg_stat_activity.datname = %s
                              AND pid <> pg_backend_pid();
                            """
                        ),
                        [db_name],
                    )
                    # Drop the database
                    cursor.execute(
                        sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name))
                    )
                    # Create the database
                    cursor.execute(
                        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                    )
                    print(f"Database '{db_name}' has been recreated.")
                    break
                else:
                    print(
                        "Invalid choice. Please enter 'U' to use or 'O' to overwrite."
                    )
        else:
            # Create the database if it doesn't exist
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
            )
            print(f"Created PostgreSQL database '{db_name}'.")

        cursor.close()
        conn.close()

    except psycopg2.Error as e:
        print(f"Error initializing PostgreSQL: {e}")
        raise e

    # Construct the database URL
    if password:
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    else:
        db_url = f"postgresql://{user}@{host}:{port}/{db_name}"
    return db_url


def run_single_study(config: dict, study_name: str, db_url: str):
    """
    Run a single Optuna study.

    Args:
        config (dict): Configuration dictionary.
        study_name (str): Name of the study.
        db_url (str): PostgreSQL connection URL.
    """
    if not config.get("optuna_logging", False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=config["seed"])

    if config["prune"]:
        epochs = config["epochs"]
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=epochs // 8, max_resource=epochs, reduction_factor=2
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=db_url,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    device_queue = queue.Queue()
    for dev in config["devices"]:
        device_queue.put(dev)

    objective_fn = create_objective(config, study_name, device_queue)
    n_trials = config["n_trials"]
    trial_durations = []

    def trial_complete_callback(study_, trial_):
        if trial_.state == TrialState.COMPLETE:
            trial_pbar.update(1)
            if trial_.datetime_start:
                duration = time.time() - trial_.datetime_start.timestamp()
                trial_durations.append(duration)
                avg_duration = sum(trial_durations) / len(trial_durations)
                remaining = n_trials - len(trial_durations)
                eta_seconds = avg_duration * remaining
                trial_pbar.set_postfix(
                    {
                        "ETA": f"{eta_seconds/60:.1f}m",
                        "LastTrial": f"{duration:.1f}s",
                    }
                )
        elif trial_.state == TrialState.PRUNED:
            trial_pbar.update(1)

    with tqdm(
        total=n_trials, desc=f"Tuning {study_name}", position=1, leave=True
    ) as trial_pbar:
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            n_jobs=len(config["devices"]),
            callbacks=[
                MaxTrialsCallback(n_trials, states=[TrialState.COMPLETE]),
                trial_complete_callback,
            ],
        )


def run_all_studies(config: dict, main_study_name: str, db_url: str):
    """
    Run all sub-studies for the given main study.

    Args:
        config (dict): Configuration dictionary.
        main_study_name (str): Name of the main study.
        db_url (str): PostgreSQL connection URL.
    """

    surrogates = config["surrogates"]
    total_sub_studies = len(surrogates)

    with tqdm(
        total=total_sub_studies, desc="Overall Surrogates", position=0, leave=True
    ) as arch_pbar:
        for i, surr in enumerate(surrogates, start=1):
            arch_name = surr["name"]
            study_name = f"{main_study_name}_{arch_name.lower()}"
            arch_pbar.set_postfix({"study": study_name})
            trials = surr.get("trials", config.get("trials", None))

            sub_config = {
                "batch_size": surr["batch_size"],
                "dataset": config["dataset"],
                "devices": config["devices"],
                "epochs": surr["epochs"],
                "n_trials": trials,
                "seed": config["seed"],
                "surrogate": {"name": arch_name},
                "optuna_params": surr["optuna_params"],
                "prune": config.get("prune", True),
                "optuna_logging": config.get("optuna_logging", False),
                "use_optimal_params": config.get("use_optimal_params", False),
            }

            run_single_study(sub_config, study_name, db_url)

            arch_pbar.update(1)
            arch_pbar.set_postfix({"done": study_name})


def initialize_optuna_database(config, study_folder_name):
    """
    Initialize the PostgreSQL server and Optuna database.

    Args:
        config (dict): Configuration dictionary.
        study_folder_name (str): Name of the study folder to use as default database name.

    Returns:
        str: PostgreSQL connection URL.
    """
    # Check if PostgreSQL server is running
    try:
        check_postgres_running(config)
        print("PostgreSQL server is running.")
    except Exception as e:
        print(str(e))
        print("Attempting to start PostgreSQL server...")
        try:
            start_postgres_server(config)
            # Wait a moment for the server to start
            time.sleep(2)
            # Re-check if it's running
            check_postgres_running(config)
            print("PostgreSQL server is now running.")
        except Exception as e_start:
            print(f"Failed to start PostgreSQL server: {e_start}")
            sys.exit(1)

    # Initialize the Optuna database
    try:
        db_url = initialize_postgres(config, study_folder_name)
    except Exception as e:
        print(str(e))
        sys.exit(1)

    return db_url


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Optuna tuning studies locally.")
    parser.add_argument(
        "--study_name",
        type=str,
        default="lotkavolterra2",
        help="Study identifier.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """Main function to run the Optuna tuning."""
    nice_print("Starting Optuna tuning")
    args = parse_arguments()
    config_path = os.path.join("optuna_runs", args.study_name, "optuna_config.yaml")

    # Derive study_folder_name from config_path
    study_folder_name = os.path.basename(os.path.dirname(config_path))

    config = load_yaml_config(config_path)

    optuna_logging = config.get("optuna_logging", False)
    if not optuna_logging:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print("Optuna logging disabled. No intermediate results will be printed.")

    # Initialize PostgreSQL server and Optuna database
    db_url = initialize_optuna_database(config, study_folder_name)

    if "surrogates" in config:
        run_all_studies(config, args.study_name, db_url)
    else:
        run_single_study(config, args.study_name, db_url)

    nice_print("Optuna tuning completed!")
