# Modalities & Studies

Modalities extend the “main” training run by stressing models under additional data regimes. Every modality toggled in `config.yaml` multiplies the number of checkpoints produced. This page explains what each block does and how it impacts training/evaluation time.

## Overview

| Modality | Configuration Keys | What it does | Training impact | Evaluation impact |
| --- | --- | --- | --- | --- |
| Interpolation | `interpolation.enabled`, `interpolation.intervals` | Removes segments of the time axis and asks the surrogate to fill them in. | One extra training per interval per surrogate. | Enables interpolation metrics/plots when `losses`/`iterative`/`compare` reference them. |
| Extrapolation | `extrapolation.enabled`, `extrapolation.cutoffs` | Trains models on truncated trajectories and evaluates beyond the cutoff. | One extra training per cutoff per surrogate. | Produces extrapolation error charts during evaluation. |
| Sparse data | `sparse.enabled`, `sparse.factors` | Down-samples observations before training to test data efficiency. | One extra training per sparsity factor. | Evaluation compares sparse-vs-dense performance. |
| Batch scaling | `batch_scaling.enabled`, `batch_scaling.sizes` | Sweeps different batch-size multipliers (floats or fractions). | One extra training per batch size (metric equals the literal batch value). | Timing/compute comparisons capture throughput scaling. |
| Uncertainty (Deep Ensembles) | `uncertainty.enabled`, `uncertainty.ensemble_size` | Trains multiple seeds of the same configuration to estimate epistemic uncertainty. | Adds `(ensemble_size - 1)` trainings per surrogate (the “main” model counts as the first member). | Unlocks uncertainty plots (e.g., catastrophic detection curves). |

All modalities share the same dataset preprocessing as the main run. Seeds are offset to keep ensembles diverse but deterministic.

## Practical guidance

- **Start narrow** — toggle one modality at a time when prototyping (interpolation is a good first candidate). Modalities multiply training duration rapidly.
- **Resume friendly** — because each modality corresponds to independent tasks, you can stop the training script mid-run and resume later; completed modalities stay finished.
- **Evaluation switches are separate** — enabling a modality does not automatically run every evaluation suite. Remember to set `losses`, `timing`, `compute`, etc. if you want the associated reports.
- **Iterative rollouts** — the `iterative` flag under evaluation is independent but often paired with extrapolation to study long-horizon drift.

See the :doc:`configuration reference </reference/configuration>` for the exact YAML schema and defaults.
