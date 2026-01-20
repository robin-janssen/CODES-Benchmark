# Getting Started

CODES Benchmark helps you compare surrogate models for coupled ODE systems by running consistent training, evaluation, and reporting pipelines. This page summarizes the minimum you need to install the project, configure a run, and validate that everything is wired correctly.

## Prerequisites

- Python 3.10 with `pip` (we recommend `uv` for faster installs, but plain pip works)
- (Optional) CUDA-capable GPU if you want to reproduce the default configurations
- Enough disk space to store downloaded datasets and the checkpoints created under `trained/` and `results/`

## Installation

We recommend [`uv`](https://docs.astral.sh/uv/) for deterministic, reproducible installs. Plain `pip` instructions are listed afterwards if you prefer a traditional virtual environment.

**uv (recommended)**

```bash
git clone https://github.com/robin-janssen/CODES-Benchmark.git
cd CODES-Benchmark
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt
```

**pip / virtualenv**

```bash
git clone https://github.com/robin-janssen/CODES-Benchmark.git
cd CODES-Benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

The editable install exposes the `codes` package locally so scripts such as `run_training.py` and `run_eval.py` can import it without extra path hacks. Installing the pinned requirements ensures feature parity with CI and the documentation build.

## Configure a run

Every benchmark—no matter how large—follows the same pattern: **hyperparameter tuning → full training → evaluation/reporting**. Before diving into the YAML knobs, consider running a tiny “sanity” training + eval on a lightweight dataset to confirm the environment, GPU drivers, and filesystem permissions all work on your machine.

All benchmark settings live in YAML files. The default `config.yaml` is a sensible starting point:

```yaml
training_id: "training_test"
surrogates: ["MultiONet", "FullyConnected"]
devices: ["cuda:0"]
dataset:
  name: "osu2008"
  log10_transform: true
  normalise: "minmax"
batch_scaling:
  enabled: false
```

1. Copy `config.yaml` or `config_full.yaml` to a new name inside the repo.
2. Update the surrogate list, dataset, devices, and optional study switches such as `interpolation`, `extrapolation`, or `uncertainty`.
3. Keep the file under version control so you can trace results.

See the [configuration reference](reference/configuration.md) for a complete list of keys, defaults, and tips.

## Run your first benchmark

A quick smoke test keeps iteration times down:

1. Duplicate `config.yaml` and set it to a **single surrogate**, **single dataset** (for example `osu2008`), and **tiny epoch/batch sizes** so the run finishes in minutes.
2. **Train** the requested surrogate (creates `trained/<training_id>`):
   ```bash
   python run_training.py --config path/to/your_config.yaml
   ```
3. **Evaluate / benchmark** everything that was trained:
   ```bash
   python run_eval.py --config path/to/your_config.yaml
   ```
4. Inspect the generated tables under `results/<training_id>` and the plots inside `plots/<training_id>`.

Every CLI script honours the `--config` flag and logs progress to the console.

## Where to go next

- [Running benchmarks](guides/running-benchmarks/index.md): deep-dive into the workflow, multi-device execution, and troubleshooting.
- [Configuration reference](reference/configuration.md): every knob explained.
- [Dataset catalog](reference/datasets.md): discover the bundled datasets and download URLs.
- [Interactive tutorial](tutorials/benchmark_quickstart.ipynb): hands-on notebook that walks through loading configs, triggering runs, and parsing outputs.
