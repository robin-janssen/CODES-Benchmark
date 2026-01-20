# Extending The Benchmark

The benchmark is designed so you can plug in new datasets or surrogate models without rewriting the orchestration code. This guide summarizes the extension points and the files you need to touch.

## Add a dataset

1. Create a folder under `datasets/<dataset_name>/` that contains `data.hdf5`. The file must expose `train`, `test`, and `val` datasets shaped as `(n_samples, n_timesteps, n_quantities)`. Optional datasets: `train_params`, `test_params`, `val_params`, and `timesteps`.
2. Register the download URL in `datasets/data_sources.yaml` if the dataset is publicly hosted.
3. Update your configuration file so `dataset.name` matches the new folder. All preprocessing options—log transforms, per-species normalization, tolerance—can remain unchanged unless your data requires special treatment.
4. Re-run `run_training.py`; the data loader will automatically pick up the new dataset after the configuration change.

## Add a surrogate model

1. Implement a class under `codes/surrogates/` that inherits from `SurrogateModel`. The `surrogates/surrogate_classes.py` module is a useful template.
2. Register the class inside `codes/surrogates/__init__.py` by updating the `surrogate_classes` mapping so the CLI can access it by name.
3. Document any model-specific settings in `config_full.yaml` and in the new docs so users know which keys to set inside the `surrogate_configs` block.
4. Train the surrogate via `run_training.py` and include it in your benchmarking config. The evaluation scripts will load checkpoints using the name you registered.

## Customize benchmark modes

- Modes such as interpolation, extrapolation, sparse training, and uncertainty ensembles are all controlled through the configuration file. When you extend the benchmark with new studies, follow the same schema: nest an `enabled` flag and provide the parameter grid.
- For reproducible sweeps, rely on the auto-generated task list. Each task stores the surrogate name, the mode, and the scalar defining that mode (interval, cutoff, sparsity factor, etc.).

## Share results

- Export the CSV summaries under `results/` or wrap the plotting helpers in a notebook (see [tutorials](../tutorials/index.md)).
- Consider contributing back reusable datasets or surrogates via pull requests so the community benefits from your extensions.
