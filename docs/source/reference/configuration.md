# Configuration Reference

All orchestration happens through YAML configuration files. The schema is purposely flat so it can be edited by hand or generated programmatically. Use this page as a lookup table while editing `config.yaml` or building custom variants.

## Top-level keys

| Key | Description |
| --- | --- |
| `training_id` | Folder name under `trained/`, `results/`, and `plots/`. Choose a unique value per experiment. |
| `surrogates` | Ordered list of surrogate model names registered in `codes.surrogates`. |
| `batch_size` | Either a single integer or a list aligned with `surrogates`. Controls the per-model training batch size. |
| `epochs` | Int or list; maps to the number of epochs per surrogate. |
| `devices` | List of device strings (`cpu`, `cuda:0`, …) used by the training workers. |
| `seed` | Seed forwarded to PyTorch, NumPy, and Python for reproducibility. |
| `verbose` | Enables additional console logs during data loading and preprocessing. |

## Dataset block

```yaml
dataset:
  name: "osu2008"
  log10_transform: true
  log10_transform_params: false
  normalise: "minmax"    # or "standardise" / "disable"
  normalise_per_species: false
  tolerance: 1e-25
  subset_factor: 1
  log_timesteps: true
  use_optimal_params: true
```

- `name`: folder inside `datasets/`.
- `normalise`: choose `"disable"` if the dataset is already scaled.
- `subset_factor`: down-samples the dataset for quick smoke tests.
- `log_timesteps`: indicates whether timesteps are logarithmically spaced, which is used when plotting or interpolating.

## Benchmark toggles

Each study type follows the same pattern: an `enabled` flag and the list/scalar that parameterizes the sweep.

```yaml
interpolation:
  enabled: true
  intervals: [2, 3, 4]
extrapolation:
  enabled: false
  cutoffs: [50, 60, 70]
sparse:
  enabled: false
  factors: [2, 4, 8]
batch_scaling:
  enabled: false
  sizes: [0.125, 0.25, 0.5]
uncertainty:
  enabled: false
  ensemble_size: 5
iterative:
  enabled: false
```

- `interpolation` / `extrapolation`: Train additional checkpoints on partial timesteps and compare roll-outs.
- `sparse`: Reduce available training samples by the listed factors.
- `batch_scaling`: Sweeps over different minibatch sizes to study throughput.
- `uncertainty`: Uses deep ensembles; expect `ensemble_size` copies per surrogate.
- `iterative`: Re-feeds predictions as inputs during evaluation to expose long-horizon drift.

## Evaluation switches

```
losses: true
gradients: true
timing: true
compute: true
compare: true
```

Turn these off to speed up `run_eval.py` when you only need a subset of analyses.

## Surrogate-specific configuration

Each surrogate can declare additional hyperparameters inside a `surrogate_configs` block:

```yaml
surrogate_configs:
  MultiONet:
    hidden_layers: [512, 512]
    activation: gelu
  LatentNeuralODE:
    latent_dim: 64
    solver: dopri5
```

Refer to the docstrings of the surrogate classes for the accepted keys. Any dict provided here will be forwarded to the class constructor via `config=model_config`.
