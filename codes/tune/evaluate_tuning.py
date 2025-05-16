import argparse
import math
import os
import re
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import psycopg2
import torch
from optuna.trial import TrialState
from psycopg2 import sql

from codes.tune import load_yaml_config
from codes.utils import nice_print


def load_loss_history(model_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load loss histories from a saved model file (.pth).
    Returns (train_loss, test_loss, n_epochs).
    """
    model_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    attributes = model_dict.get("attributes", {})
    train_loss = (
        np.array(attributes.get("train_loss"))
        if attributes.get("train_loss") is not None
        else None
    )
    test_loss = (
        np.array(attributes.get("test_loss"))
        if attributes.get("test_loss") is not None
        else None
    )
    n_epochs = attributes.get(
        "n_epochs", len(train_loss) if train_loss is not None else 0
    )
    return train_loss, test_loss, n_epochs


def plot_losses(
    loss_histories, epochs, labels, title="Losses", save=False, out_dir=None, mode=""
):
    """
    Plot multiple loss trajectories on a log scale.
    """
    valid = [loss for loss in loss_histories if loss is not None and loss.size]
    if not valid:
        print("No valid loss arrays; skipping.")
        return
    max_len = max(len(loss) for loss in valid)
    # Determine y-limits excluding initial 2%
    mins, maxs = [], []
    for loss in valid:
        start = int(len(loss) * 0.02)
        vals = loss[start:][loss[start:] > 0]
        if vals.size:
            mins.append(vals.min())
            maxs.append(vals.max())
    ymin = min(mins) if mins else 1e-8
    ymax = max(maxs) if maxs else 1.0
    # colors = plt.cm.magma(np.linspace(0.15, 0.85, len(loss_histories)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_histories)))

    plt.figure(figsize=(6, 4))
    for loss, lab in zip(loss_histories, labels):
        plt.plot(np.arange(len(loss)), loss, label=lab, color=colors[labels.index(lab)])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.xlim(0, max_len)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.legend()
    if save and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"losses_{mode}.png"), dpi=300)
        print(f"Saved plot to {out_dir}")
    plt.close()


def get_best_trials(study: optuna.Study, top_n: int) -> list[int]:
    """
    Return trial numbers of best 'top_n' trials.
    Single-objective: lowest value.
    Multi-objective: Pareto front; if fewer than top_n, pad with closest-to-origin among the rest.
    """
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        return []
    # Single-objective
    if len(study.directions) == 1:
        sorted_trials = sorted(completed, key=lambda t: t.value)
        return [t.number for t in sorted_trials[:top_n]]

    # Multi-objective
    def dominated(a, b):
        return all(bi <= ai for ai, bi in zip(a, b)) and any(
            bi < ai for ai, bi in zip(a, b)
        )

    # Pareto front
    pareto = [
        t
        for t in completed
        if not any(dominated(t.values, u.values) for u in completed if u != t)
    ]
    # Compute distances to origin for all trials
    dist_all = [(t.number, math.hypot(*t.values)) for t in completed]
    # Sort by distance
    dist_all.sort(key=lambda x: x[1])
    # Select top_n by prioritizing Pareto, then padding from sorted distances
    selected = []
    # First, add pareto points in any order (or by distance)
    pareto_dists = [(t.number, math.hypot(*t.values)) for t in pareto]
    pareto_dists.sort(key=lambda x: x[1])
    for num, _ in pareto_dists:
        if len(selected) < top_n:
            selected.append(num)
    # Then pad with nearest non-pareto
    if len(selected) < top_n:
        for num, _ in dist_all:
            if num in selected:
                continue
            selected.append(num)
            if len(selected) >= top_n:
                break
    return selected


def evaluate_tuning(study_prefix: str, top_n: int = 10) -> None:
    """
    For all surrogate studies named '<study_prefix>_<surrogate>',
    plot the top_n test-loss trajectories.
    """
    # Load prefix config
    config_path = os.path.join("tuned", study_prefix, "optuna_config.yaml")
    if not os.path.exists(config_path):
        print(f"Config missing: {config_path}")
        return
    config = load_yaml_config(config_path)

    # Ensure Postgres server
    pg = config.get("postgres_config", {})
    try:
        psycopg2.connect(
            dbname="postgres",
            user=pg.get("user"),
            password=pg.get("password"),
            host=pg.get("host"),
            port=pg.get("port"),
            connect_timeout=5,
        ).close()
    except Exception:
        pg_data = pg.get("data_dir", os.path.expanduser("~/postgres/data"))
        pg_ctl = os.path.join(
            pg.get("database_folder", os.path.expanduser("~/postgres")), "bin", "pg_ctl"
        )
        subprocess.run(
            [pg_ctl, "-D", pg_data, "-l", pg.get("log_file", "/tmp/pg.log"), "start"],
            check=True,
        )
        time.sleep(2)
    # Create DB if absent
    conn = psycopg2.connect(
        dbname="postgres",
        user=pg.get("user"),
        password=pg.get("password"),
        host=pg.get("host"),
        port=pg.get("port"),
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [study_prefix])
    if not cur.fetchone():
        cur.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(study_prefix)))
    cur.close()
    conn.close()

    # Build storage URL (hard-coded prefix)
    db_prefix = "postgresql://optuna_user:your_password@localhost:5432/"
    storage_url = f"{db_prefix}{study_prefix}"

    # Discover all studies with this prefix
    from optuna.study import get_all_study_summaries

    summaries = get_all_study_summaries(storage=storage_url)
    study_names = [
        s.study_name for s in summaries if s.study_name.startswith(f"{study_prefix}_")
    ]
    if not study_names:
        print(f"No studies found with prefix '{study_prefix}_' in {storage_url}")
        return

        # Output directory: directly in main study folder
    save_dir = os.path.join("tuned", study_prefix)
    os.makedirs(save_dir, exist_ok=True)

    # Loop over each surrogate study
    for full_name in study_names:
        suffix = full_name[len(study_prefix) + 1 :]
        print(f"--- Evaluating study {full_name} -> surrogate '{suffix}' ---")
        try:
            study = optuna.load_study(study_name=full_name, storage=storage_url)
        except KeyError:
            print(f"Could not load study '{full_name}'")
            continue

        best = get_best_trials(study, top_n)
        if not best:
            print(f"No completed trials in {full_name}")
            continue

        # Plotting
        # Resolve surrogate models folder (case-insensitive)
        models_root = os.path.join("tuned", study_prefix, "models")
        if not os.path.isdir(models_root):
            print(f"Models root not found: {models_root}")
            continue
        available = [
            d
            for d in os.listdir(models_root)
            if os.path.isdir(os.path.join(models_root, d))
        ]
        match = next((d for d in available if d.lower() == suffix.lower()), None)
        if match is None:
            print(
                f"Models folder for surrogate '{suffix}' not found in {models_root}. Available: {available}"
            )
            continue
        models_dir = os.path.join(models_root, match)

        # Collect top trials' loss histories
        records = []
        epochs = None
        for fname in os.listdir(models_dir):
            if not fname.endswith(".pth"):
                continue
            m = re.search(r"_(\d+)\.pth$", fname)
            if not m:
                continue
            tnum = int(m.group(1))
            if tnum not in best:
                continue
            path = os.path.join(models_dir, fname)
            train, test, ep = load_loss_history(path)
            if test is None:
                continue
            records.append((tnum, test))
            epochs = ep
        if not records:
            print(f"No top-{top_n} trials for surrogate '{suffix}'")
            continue
        records.sort(key=lambda x: best.index(x[0]))
        tnums, losses = zip(*records)
        labels = [f"Trial {n}" for n in tnums]
        out_dir = save_dir
        plot_losses(
            list(losses),
            epochs,
            labels,
            title=f"{suffix} Top-{top_n}",
            save=True,
            out_dir=out_dir,
            mode=suffix,
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot top-N tuning losses for all surrogate studies."
    )
    p.add_argument(
        "--study_name",
        type=str,
        required=True,
        help="Main study prefix (e.g. lvparams5)",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of top trials to plot per surrogate",
    )
    return p.parse_args()


def main():
    args = parse_args()
    evaluate_tuning(args.study_name, args.top_n)


if __name__ == "__main__":
    nice_print("Starting evaluation of tuning results")
    main()
    nice_print("Done.")
