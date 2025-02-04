import queue

import optuna
from mpi4py import MPI
from optuna.pruners import HyperbandPruner, NopPruner
from optuna.samplers import TPESampler
from tqdm import tqdm

# Message tags
REQUEST_TRIAL = 1
SEND_TRIAL_TOKEN = 2
NO_TRIAL = 3
TRIAL_DONE = 4


def master_mpi_optuna(config: dict, study_name: str, db_url: str):
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    n_trials = config["n_trials"]
    completed_trials = 0
    sampler = TPESampler(seed=config["seed"])
    if config["prune"]:
        pruner = HyperbandPruner(
            min_resource=config["epochs"] // 8,
            max_resource=config["epochs"],
            reduction_factor=2,
        )
    else:
        pruner = NopPruner()
    try:
        study = optuna.load_study(study_name=study_name, storage=db_url)
    except Exception:
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=db_url,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )
    progress_bar = tqdm(
        total=n_trials, desc=f"Tuning {study_name}", position=1, leave=True
    )
    while completed_trials < n_trials:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        src = status.Get_source()
        tag = status.Get_tag()
        if tag == REQUEST_TRIAL:
            if completed_trials < n_trials:
                comm.send("go", dest=src, tag=SEND_TRIAL_TOKEN)
            else:
                comm.send(None, dest=src, tag=NO_TRIAL)
        elif tag == TRIAL_DONE:
            completed_trials += 1
            progress_bar.update(1)
    progress_bar.close()
    for r in range(1, comm.Get_size()):
        comm.send(None, dest=r, tag=NO_TRIAL)


def worker_mpi_optuna(config: dict, study_name: str, db_url: str):
    print("Worker started")
    from codes.tune.optuna_fcts import create_objective

    comm = MPI.COMM_WORLD
    # Wait for rank 0 to finish study creation/migration.
    comm.Barrier()
    # Now load the study.
    try:
        study = optuna.load_study(study_name=study_name, storage=db_url)
    except Exception as e:
        import time

        time.sleep(1)
        study = optuna.load_study(study_name=study_name, storage=db_url)
    device_queue = queue.Queue()
    for dev in config["devices"]:
        device_queue.put(dev)
    while True:
        comm.send(None, dest=0, tag=REQUEST_TRIAL)
        token = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())
        if token is None:
            break
        objective = create_objective(config, study_name, device_queue)
        study.optimize(objective, n_trials=1, n_jobs=1)
        comm.send("done", dest=0, tag=TRIAL_DONE)


def run_mpi_optuna_tuning(config: dict, study_name: str, db_url: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        master_mpi_optuna(config, study_name, db_url)
    else:
        worker_mpi_optuna(config, study_name, db_url)
