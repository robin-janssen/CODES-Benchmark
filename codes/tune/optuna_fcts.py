import os
import queue
import subprocess
import sys
import time
from distutils.util import strtobool

import optuna
import psycopg2
import torch
import torch.nn as nn
import yaml
from psycopg2 import sql
from tqdm import tqdm

from codes.benchmark.bench_utils import get_model_config, get_surrogate
from codes.utils import check_and_load_data, make_description, set_random_seeds
from codes.utils.data_utils import download_data, get_data_subset


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_activation_function(name: str) -> nn.Module:
    activation_functions = {
        "relu": nn.ReLU(),
        "leakyrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "softplus": nn.Softplus(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity(),
        "elu": nn.ELU(),
    }
    return activation_functions[name.lower()]


def make_optuna_params(trial: optuna.Trial, optuna_params: dict) -> dict:
    suggested_params = {}
    for param_name, param_options in optuna_params.items():
        if param_options["type"] == "int":
            suggested_params[param_name] = trial.suggest_int(
                param_name, param_options["low"], param_options["high"]
            )
        elif param_options["type"] == "float":
            suggested_params[param_name] = trial.suggest_float(
                param_name,
                param_options["low"],
                param_options["high"],
                log=param_options.get("log", False),
            )
        elif param_options["type"] == "categorical":
            suggested_params[param_name] = trial.suggest_categorical(
                param_name, param_options["choices"]
            )
    return suggested_params


def create_objective(
    config: dict, study_name: str, device_queue: queue.Queue
) -> callable:
    def objective(trial):
        device = device_queue.get()
        try:
            try:
                return training_run(trial, device, config, study_name)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                msg = repr(e).strip() or "CUDA Out of Memory."
                print(f"Trial {trial.number} failed due to: {msg}")
                trial.set_user_attr("exception", msg)
                raise optuna.TrialPruned(f"OOM error in trial {trial.number}")
            except optuna.TrialPruned as e:
                trial.set_user_attr("exception", repr(e).strip())
                raise
            except Exception as e:
                torch.cuda.empty_cache()
                msg = repr(e).strip() or "Unknown error occurred."
                print(f"Trial {trial.number} failed due to an unexpected error: {msg}")
                trial.set_user_attr("exception", msg)
                raise optuna.TrialPruned(f"Error in trial {trial.number}: {msg}")
        finally:
            device_queue.put(device)

    return objective


def training_run(
    trial: optuna.Trial, device: str, config: dict, study_name: str
) -> float:
    download_data(config["dataset"]["name"])
    train_data, test_data, val_data, timesteps, _, data_params, _ = check_and_load_data(
        config["dataset"]["name"],
        verbose=False,
        log=config["dataset"]["log10_transform"],
        normalisation_mode=config["dataset"]["normalise"],
    )
    subset_factor = config["dataset"].get("subset_factor", 1)
    train_data, test_data, timesteps = get_data_subset(
        train_data, test_data, timesteps, "sparse", subset_factor
    )
    set_random_seeds(config["seed"], device=device)
    surr_name = config["surrogate"]["name"]
    suggested_params = make_optuna_params(trial, config["optuna_params"])
    for key, val in suggested_params.items():
        if "activation" in key:
            suggested_params[key] = get_activation_function(val)
        if "ode_tanh_reg" in key:
            suggested_params[key] = bool(strtobool(val))
    n_timesteps = train_data.shape[1]
    n_chemicals = train_data.shape[2]
    surrogate_class = get_surrogate(surr_name)
    model_config = get_model_config(surr_name, config)
    model_config.update(suggested_params)
    model = surrogate_class(device, n_chemicals, n_timesteps, model_config)
    model.optuna_trial = trial
    model.trial_update_epochs = 10
    train_loader, test_loader, _ = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=val_data,
        timesteps=timesteps,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    description = make_description("Optuna", device, str(trial.number), surr_name)
    pos = config["devices"].index(device) + 2 if device in config["devices"] else 2
    model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config["epochs"],
        position=pos,
        description=description,
    )
    criterion = torch.nn.MSELoss()
    preds, targets = model.predict(test_loader)
    loss = criterion(preds, targets).item()
    sname, _ = study_name.split("_")
    savepath = os.path.join("optuna_runs", sname, "models")
    os.makedirs(savepath, exist_ok=True)
    model_name = f"{surr_name.lower()}_{trial.number}"
    model.save(model_name=model_name, base_dir="", training_id=savepath)
    return loss


def run_single_study(config: dict, study_name: str, db_url: str):
    if not config.get("optuna_logging", False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    from optuna.pruners import HyperbandPruner, NopPruner
    from optuna.samplers import TPESampler

    sampler = TPESampler(seed=config["seed"])
    if config["prune"]:
        epochs = config["epochs"]
        pruner = HyperbandPruner(
            min_resource=epochs // 8, max_resource=epochs, reduction_factor=2
        )
    else:
        pruner = NopPruner()
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
        if trial_.state == optuna.trial.TrialState.COMPLETE:
            trial_pbar.update(1)
            if trial_.datetime_start:
                duration = time.time() - trial_.datetime_start.timestamp()
                trial_durations.append(duration)
                avg_duration = sum(trial_durations) / len(trial_durations)
                remaining = n_trials - len(trial_durations)
                eta_seconds = avg_duration * remaining
                trial_pbar.set_postfix(
                    {"ETA": f"{eta_seconds/60:.1f}m", "LastTrial": f"{duration:.1f}s"}
                )
        elif trial_.state == optuna.trial.TrialState.PRUNED:
            trial_pbar.update(1)

    with tqdm(
        total=n_trials, desc=f"Tuning {study_name}", position=1, leave=True
    ) as trial_pbar:
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            n_jobs=len(config["devices"]),
            callbacks=[
                optuna.study.MaxTrialsCallback(
                    n_trials, states=[optuna.trial.TrialState.COMPLETE]
                ),
                trial_complete_callback,
            ],
        )


def run_all_studies(config: dict, main_study_name: str, db_url: str):
    surrogates = config["surrogates"]
    total_sub_studies = len(surrogates)
    from tqdm import tqdm

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


########################
# PostgreSQL functions
########################


def check_postgres_running(config):
    db_config = config.get("postgres_config", {})
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "optuna_user")
    password = db_config.get("password", "")
    try:
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
    db_config = config.get("postgres_config", {})
    data_dir = db_config.get("data_dir", os.path.expanduser("~/postgres/data"))
    log_file = db_config.get("log_file", os.path.expanduser("~/postgres/logfile"))
    database_folder = db_config.get("database_folder", os.path.expanduser("~/postgres"))
    pg_ctl_path = os.path.join(database_folder, "bin", "pg_ctl")
    if not os.path.exists(data_dir):
        raise Exception(f"PostgreSQL data directory '{data_dir}' does not exist.")
    if not os.path.exists(pg_ctl_path):
        raise Exception(f"pg_ctl not found at '{pg_ctl_path}'.")
    try:
        print("Starting PostgreSQL server...")
        subprocess.run(
            [pg_ctl_path, "-D", data_dir, "-l", log_file, "start"], check=True
        )
        print("PostgreSQL server started successfully.")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to start PostgreSQL server: {e}")


def initialize_postgres(config, study_folder_name):
    try:
        from mpi4py import MPI

        mpi_available = True
    except ImportError:
        mpi_available = False
    rank = 0
    size = 1
    if mpi_available:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    db_config = config.get("postgres_config", {})
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "optuna_user")
    password = db_config.get("password", "")
    db_name = db_config.get("db_name", study_folder_name)

    if rank == 0:
        conn = psycopg2.connect(
            dbname="postgres", user=user, password=password, host=host, port=port
        )
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(
            sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [db_name]
        )
        exists = cursor.fetchone() is not None
    else:
        exists = None

    if mpi_available and size > 1:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        exists = comm.bcast(exists, root=0)

    if exists:
        decision = None
        if rank == 0:
            while True:
                decision = (
                    input(
                        f"Database '{db_name}' exists. Do you want to (U)se it or (O)verwrite it? [U/O]: "
                    )
                    .strip()
                    .lower()
                )
                if decision in ("u", "o"):
                    break
                print("Invalid choice. Please enter 'U' or 'O'.")
        if mpi_available and size > 1:
            decision = comm.bcast(decision, root=0)
        if decision == "o":
            if rank == 0:
                print(f"Overwriting database '{db_name}'.")
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
                cursor.execute(
                    sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name))
                )
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                )
                print(f"Database '{db_name}' has been recreated.")
        else:
            if rank == 0:
                print(f"Using existing database '{db_name}'.")
    else:
        if rank == 0:
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
            )
            print(f"Created PostgreSQL database '{db_name}'.")
    if rank == 0:
        cursor.close()
        conn.close()
    if password:
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    else:
        db_url = f"postgresql://{user}@{host}:{port}/{db_name}"
        print(f"Database URL: {db_url}")
    return db_url


def initialize_optuna_database(config, study_folder_name):
    try:
        check_postgres_running(config)
        print("PostgreSQL server is running.")
    except Exception as e:
        print(str(e))
        print("Attempting to start PostgreSQL server...")
        try:
            start_postgres_server(config)
            time.sleep(2)
            check_postgres_running(config)
            print("PostgreSQL server is now running.")
        except Exception as e_start:
            print(f"Failed to start PostgreSQL server: {e_start}")
            sys.exit(1)
    try:
        db_url = initialize_postgres(config, study_folder_name)
    except Exception as e:
        print(str(e))
        sys.exit(1)
    return db_url
