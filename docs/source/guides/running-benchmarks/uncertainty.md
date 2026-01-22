# Uncertainty Quantification

Surrogates occasionally make large errors even on familiar datasets. CODES provides optional deep ensembles (modalities → `uncertainty`) so you can quantify confidence and fall back to the numerical solver when needed.

## Deep Ensemble basics

- Enable by setting `uncertainty.enabled: true` and `ensemble_size: M` in your training config.
- Training seeds are offset per ensemble member so each model explores a different region of the loss landscape.
- During evaluation, predictions are averaged across members; the standard deviation across members becomes the **predicted uncertainty**.

## What it buys you

1. **Potential accuracy gains** — averaging multiple members can reduce variance and improve mean performance, especially on noisy datasets.
2. **Reliability signals** — high predicted uncertainty correlates with large actual errors, enabling selective trust or fallback strategies.

## Costs

- Training time scales linearly with `M` because each ensemble member is a separate task (see the modalities guide).
- Inference time also scales with `M` when you request uncertainty-aware predictions, though most CODES surrogates have lightweight forward passes so the overhead is modest.

## Evaluation hooks

- Enable the `uncertainty` modality plus `losses`/`compare` evaluation switches to generate catastrophic-detection plots, uncertainty-vs-error correlations, and ensemble statistics.
- Use the outputs to design thresholds (e.g., “re-run the numerical solver when predicted uncertainty exceeds X dex”).

Deep ensembles are currently the default UQ mechanism, but the interface leaves room for future methods (e.g., MC Dropout, SWAG). Whatever approach you adopt, test it in the evaluation stage to ensure uncertainty signals align with actual errors.
