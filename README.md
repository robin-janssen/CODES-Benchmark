# CODES Benchmark

[![codecov](https://codecov.io/github/robin-janssen/CODES-Benchmark/branch/main/graph/badge.svg?token=TNF9ISCAJK)](https://codecov.io/github/robin-janssen/CODES-Benchmark) ![Static Badge](https://img.shields.io/badge/license-GPLv3-blue) ![Static Badge](https://img.shields.io/badge/NeurIPS-2024-green)

🎉 Accepted to the ML4PS workshop @ NeurIPS 2024

Benchmark coupled ODE surrogate models on curated datasets with reproducible training, evaluation, and visualization pipelines. CODES helps you answer: *Which surrogate architecture fits my data, accuracy target, and runtime budget?*

## What you get

- Baseline surrogates (MultiONet, FullyConnected, LatentNeuralODE, LatentPoly) with configurable hyperparameters
- Rich datasets spanning chemistry, astrophysics, and dynamical systems
- Optional studies for interpolation/extrapolation, sparse data regimes, uncertainty estimation, and batch scaling
- Automated reporting: accuracy tables, resource usage, gradient analyses, and dozens of diagnostic plots

## Two-minute quickstart

```bash
git clone https://github.com/robin-janssen/CODES-Benchmark.git
cd CODES-Benchmark
python -m venv .venv && source .venv/bin/activate  # optional
pip install -e .
pip install -r requirements.txt
python run_training.py --config config.yaml
python run_eval.py --config config.yaml
```

Outputs land in `trained/<training_id>`, `results/<training_id>`, and `plots/<training_id>`. Copy `config.yaml` to a new file to customize datasets, surrogates, and benchmark modes.

## Documentation

- [Main docs & tutorials](https://robin-janssen.github.io/CODES-Benchmark/)
- [API reference (Sphinx)](https://robin-janssen.github.io/CODES-Benchmark/modules.html)
- [Paper on arXiv](https://arxiv.org/abs/2410.20886)

The GitHub Pages site now hosts the narrative guides, configuration reference, and interactive notebooks alongside the generated API docs.

## Repository map

| Path | Purpose |
| --- | --- |
| `config.yaml`, `config_full.yaml` | Ready-to-edit benchmark configurations |
| `datasets/` | Bundled datasets + download helper (`data_sources.yaml`) |
| `codes/` | Python package with surrogates, training, tuning, and benchmarking utilities |
| `run_training.py`, `run_eval.py`, `run_tuning.py` | CLI entry points for the main workflows |
| `docs/` | Sphinx project powering the GitHub Pages site (guides + API reference) |
| `scripts/` | Convenience tooling (dataset downloads, experiment utilities) |

## Contributing

Pull requests are welcome! Please include documentation updates, add or update tests when you touch executable code, and run:

```bash
uv pip install --group dev
pytest
sphinx-build -b html docs/ docs/_build
```

If you publish a new surrogate or dataset, document it under `docs/guides` / `docs/reference` so users can adopt it quickly. For questions, open an issue on GitHub.
