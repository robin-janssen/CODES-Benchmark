# Training Surrogates

`run_training.py` orchestrates the heavy lifting: it loads your config, expands optional studies, schedules work across devices, and stores checkpoints under `trained/<training_id>/`.

## Queue layout

1. Start from `config.yaml`/`config_full.yaml` and set a distinct `training_id`.
2. List one or more `surrogates`. Order matters—paired lists such as `batch_size` and `epochs` are position-dependent.
3. Toggle studies per surrogate (interpolation/extrapolation/sparse/batch_scaling/uncertainty/iterative). Each switch multiplies the queue, so keep the cartesian explosion in mind.

## Launch a run

```bash
python run_training.py --config configs/demo_trained.yaml --devices cuda:0 cuda:1
```

- The CLI spins up one process per device and streams tasks from a shared queue. Use `--devices cpu` for sequential runs.
- Intermediate artifacts live in `trained/<training_id>/<surrogate>/<study>/`. Logs capture stdout/stderr so you can audit failures.
- Task metadata is persisted as JSON; interrupted runs can resume and will skip finished tasks automatically.

## Practical tips

- Monitor GPU utilization with `watch -n 1 nvidia-smi` or `torchrun` logs to size batch sizes correctly.
- Use short `max_epochs` for early profiling; bump them only once throughput looks healthy.
- Record `config.yaml` in version control alongside each `training_id` so evaluations remain reproducible.
