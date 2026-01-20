# Evaluation & Reporting

Once training finishes, `run_eval.py` reloads the checkpoints, runs accuracy/timing/uncertainty studies, and writes consolidated artifacts under `results/` and `plots/`.

## Kick off evaluations

```bash
python run_eval.py --config configs/demo_trained.yaml --modes accuracy timing compute
```

- Pass the same config that drove training so dataset transforms, scaling, and study toggles stay in sync.
- `--modes` filters which benchmark suites run; omit it to execute everything.
- Evaluations can run on a mix of CPU/GPU depending on the surrogate—set `CUDA_VISIBLE_DEVICES` or the config’s `devices` list to control placement.

## Outputs

- `results/<training_id>/<surrogate>/` stores YAML + CSV metrics (error percentiles, runtime stats, parameter counts, etc.).
- `plots/<training_id>/` includes comparison charts (loss curves, catastrophic-detection curves, error heatmaps).
- Aggregated CSVs from `make_comparison_csv` make it easy to sweep across surrogates or datasets in a spreadsheet.

## Inspecting failures

- `run_eval.py` calls `check_benchmark` to verify that configuration, checkpoints, and dataset metadata match. Resolve those mismatches before re-running the evaluation.
- Missing checkpoints or corrupt HDF5 files are surfaced with actionable tracebacks—fix the root cause and rerun; completed surrogates are cached so you pay only for the failed slices.
- For notebook-driven debugging, open `tutorials/benchmark_quickstart.ipynb` and load the generated CSVs/plots inline.
