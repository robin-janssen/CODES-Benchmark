# Hyperparameter Tuning

Tuning gives each surrogate a fair shot before you invest cycles in full training runs. The entry point is `run_tuning.py`, which wraps [Optuna](https://optuna.org/) studies and stores everything under `tuned/<training_id>`.

## Define a study

```yaml
training_id: "demo"
study:
  enable: true
  n_trials: 25
  sampler: "tpe"
  timeout_minutes: 60
surrogates:
  - "MultiONet"
dataset:
  name: "osu2008"
```

- Set a unique `training_id` so each study gets its own folder and SQLite database.
- Constrain `n_trials`/`timeout_minutes` to match your hardware budget. You can always resume a study to accumulate more evidence later.
- Most surrogates expose a dedicated `*_config.py` with Optuna search spaces—extend those to tune new parameters.

## Run the tuner

```bash
python run_tuning.py --config configs/demo.yaml --devices cuda:0
```

Key behaviours:

- One worker per listed device; pass `cpu` for debugging the search logic.
- Checkpoints for the best trial land in `tuned/<training_id>/<surrogate>/best`. Additional trial metadata stays inside the Optuna database.
- You can resume a study by re-running the same command—Optuna skips finished trials and continues sampling.

## Promote tuned configs

Use `scripts/promote_tuning.py` (or copy the reported hyperparameters manually) to update your training config:

```bash
python scripts/promote_tuning.py \\
  --training-id demo \\
  --surrogate MultiONet \\
  --output configs/demo_trained.yaml
```

The emitted YAML mirrors the `surrogates.<name>.config` section consumed by `run_training.py`, so you can drop it straight into the training phase.
