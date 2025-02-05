import os

import torch
from mpi4py import MPI
from tqdm import tqdm

from codes.train.train_fcts import train_and_save_model
from codes.utils import save_task_list

REQUEST_TASK = 1
SEND_TASK = 2
NO_TASK = 3
TASK_DONE = 4


def master_mpi(tasks, size, task_list_filepath):
    comm = MPI.COMM_WORLD
    status = MPI.Status()

    # Enumerate tasks so each gets a unique ID.
    indexed_tasks = list(enumerate(tasks))  # each element is (task_id, task)
    total_tasks = len(indexed_tasks)
    next_index = 0
    completed_ids = set()

    progress_bar = tqdm(total=total_tasks, desc="Training Progress")

    while len(completed_ids) < total_tasks:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == REQUEST_TASK:
            if next_index < total_tasks:
                assignment = indexed_tasks[next_index]  # (task_id, task)
                next_index += 1
                comm.send(assignment, dest=source, tag=SEND_TASK)
            else:
                comm.send(None, dest=source, tag=NO_TASK)
        elif tag == TASK_DONE:
            finished_task_id = data  # worker sends back the task ID it finished
            if finished_task_id not in completed_ids:
                completed_ids.add(finished_task_id)
                progress_bar.update(1)
                pending = [
                    task for (i, task) in indexed_tasks if i not in completed_ids
                ]
                save_task_list(pending, task_list_filepath)
    progress_bar.close()
    for rank in range(1, size):
        comm.send(None, dest=rank, tag=NO_TASK)


def worker_mpi(rank):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # device_count = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    print(f"Worker {rank} on device {local_rank}")
    # local_rank = (rank - 1) % device_count
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    while True:
        comm.send(None, dest=0, tag=REQUEST_TASK)
        assignment = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())
        if assignment is None:
            break
        task_id, task = assignment
        try:
            train_and_save_model(
                surr_name=task[0],
                mode=task[1],
                metric=task[2],
                training_id=task[3],
                seed=task[4],
                epochs=task[5],
                device=device,
                position=rank,
                threadlock=None,
                worker_id=rank,
            )
            comm.send(task_id, dest=0, tag=TASK_DONE)
        except Exception as e:
            print(f"Exception in worker {rank} for task {task}: {e}")
            import traceback

            print(traceback.format_exc())
            comm.send(task_id, dest=0, tag=TASK_DONE)
