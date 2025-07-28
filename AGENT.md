# CODES Benchmark - Agent Guide

## Build/Test Commands
- Install dependencies: `poetry install` or `pip install -r requirements.txt`
- Run tests: `pytest test/` or `python -m pytest test/`
- Run single test: `pytest test/test_data.py::test_function_name`
- Run training: `python run_training.py --config config.yaml`
- Run evaluation: `python run_eval.py --config config.yaml`
- Run hyperparameter tuning: `python run_tuning.py --config config.yaml`

## Architecture & Structure
- `codes/` - Main package with 4 modules: benchmark, surrogates, train, tune, utils
- `codes/surrogates/` - Neural network surrogate models (NN, DeepONet, NeuralODE, Polynomial)
- `codes/benchmark/` - Core benchmarking logic and evaluation metrics
- `codes/train/` - Parallel/sequential training infrastructure with task queues
- `codes/utils/` - Data handling, config management, progress bars, seeding
- `datasets/` - Training data in HDF5 format, `trained/` - Model checkpoints, `results/` - Benchmark outputs
- Uses PyTorch, torchode for ODE solving, Optuna for hyperparameter tuning, PostgreSQL for study storage

## Code Style & Conventions
- Python 3.10+, type hints required (e.g., `str | None`, `int | None`)
- Use `__all__` in `__init__.py` files for explicit exports
- Function docstrings with Args/Returns sections
- Classes use PascalCase, functions/variables use snake_case
- Threading with explicit locks, use DummyLock() for single-threaded contexts
- Config-driven architecture with YAML files, use `read_yaml_config()` utility
