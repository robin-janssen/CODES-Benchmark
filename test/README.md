# CODES Benchmark Test Suite

This directory contains comprehensive unit tests for the CODES benchmark framework, focusing on surrogate models and datasets.

## Test Files

### Core Test Modules

- **`test_surrogate_models.py`** - Comprehensive tests for all surrogate model implementations
  - Tests AbstractSurrogateModel interface compliance
  - Tests model initialization, training, prediction, and save/load functionality
  - Tests all registered surrogate model classes (FCNN, DeepONet, LatentNeuralODE, LatentPolynomial)

- **`test_datasets.py`** - Comprehensive tests for dataset functionality
  - Tests dataset loading from local and remote sources
  - Tests data normalization and preprocessing
  - Tests dataset creation and validation
  - Tests data_sources.yaml configuration

### Legacy Test Files

- **`test_data.py`** - Legacy data loading tests (kept for compatibility)
- **`test_surrogates.py`** - Legacy surrogate model tests (kept for compatibility)
- **`test_run.py`** - Integration tests for training/evaluation pipelines

### Configuration

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`README.md`** - This documentation file

## Running Tests

### Run All Tests
```bash
pytest test/
```

### Run Specific Test Modules
```bash
# Test surrogate models only
pytest test/test_surrogate_models.py

# Test datasets only
pytest test/test_datasets.py

# Test legacy functionality
pytest test/test_data.py test/test_surrogates.py
```

### Run Tests with Specific Markers
```bash
# Skip slow tests
pytest test/ -m "not slow"

# Skip download tests (tests that require internet)
pytest test/ -m "not download"

# Run only GPU tests (if GPU available)
pytest test/ -m "gpu"
```

### Run Tests with Coverage
```bash
pytest test/ --cov=codes --cov-report=html
```

### Run Tests in Parallel (if pytest-xdist installed)
```bash
pytest test/ -n auto
```

## Test Structure

### Surrogate Model Tests

The surrogate model tests are organized into several test classes:

1. **TestAbstractSurrogateModelInterface** - Tests the registry system and interface compliance
2. **TestSurrogateModelInitialization** - Tests model initialization and basic attributes
3. **TestDataPreparation** - Tests data loading and dataloader creation
4. **TestForwardPass** - Tests model forward pass functionality
5. **TestTraining** - Tests model training functionality
6. **TestPrediction** - Tests model prediction functionality
7. **TestSaveLoad** - Tests model serialization and deserialization
8. **TestDenormalization** - Tests data denormalization functionality
9. **TestOptimizer** - Tests optimizer and scheduler setup
10. **TestProgressBar** - Tests progress bar functionality

### Dataset Tests

The dataset tests are organized into several test classes:

1. **TestDataSourcesYaml** - Tests data_sources.yaml configuration
2. **TestDownloadData** - Tests dataset downloading functionality
3. **TestLocalDatasets** - Tests loading of locally available datasets
4. **TestCheckAndLoadData** - Tests the main data loading function
5. **TestCreateDataset** - Tests dataset creation functionality
6. **TestNormalizeData** - Tests data normalization functionality
7. **TestDatasetError** - Tests error handling
8. **TestIntegration** - Integration tests combining multiple operations

## Test Configuration

### Fixtures

The test suite uses several shared fixtures defined in `conftest.py`:

- `device` - Provides the device (CPU/GPU) to use for testing
- `test_constants` - Provides test constants (dimensions, batch sizes, etc.)
- `random_seed` - Ensures reproducible test results
- `temp_dir` - Provides temporary directories for file operations
- `sample_3d_data` - Provides sample training/test/validation data
- `sample_parameters` - Provides sample parameter arrays
- `mock_normalisation` - Provides mock normalization parameters

### Markers

The test suite uses custom pytest markers:

- `@pytest.mark.slow` - For tests that take longer to run
- `@pytest.mark.download` - For tests that require internet access
- `@pytest.mark.gpu` - For tests that require GPU

### Parameterization

Many tests are parameterized to run across:
- All registered surrogate model classes
- All available datasets (local and remote)
- Different normalization modes
- Different configuration options

## Test Coverage

The test suite aims for comprehensive coverage of:

### Surrogate Models
- ✅ Model initialization and configuration
- ✅ Data preparation and dataloader creation
- ✅ Forward pass functionality
- ✅ Training loop execution
- ✅ Prediction and evaluation
- ✅ Model serialization (save/load)
- ✅ Data denormalization
- ✅ Progress tracking and optimization
- ✅ Interface compliance with AbstractSurrogateModel

### Datasets
- ✅ Data loading from HDF5 files
- ✅ Dataset downloading from remote sources
- ✅ Data validation and structure checking
- ✅ Data normalization (minmax, standardization)
- ✅ Dataset creation and export
- ✅ Parameter handling
- ✅ Error handling and edge cases
- ✅ Integration workflows

## Development Guidelines

### Adding New Tests

1. **For new surrogate models**: Add tests to `test_surrogate_models.py` or create model-specific test files
2. **For new dataset functionality**: Add tests to `test_datasets.py`
3. **For integration tests**: Add to existing integration test classes or create new ones

### Test Naming Convention

- Test functions should start with `test_`
- Test classes should start with `Test`
- Use descriptive names that clearly indicate what is being tested

### Assertion Guidelines

- Use descriptive assertion messages
- Test both positive and negative cases
- Use appropriate pytest features (parametrize, fixtures, markers)
- Keep tests focused and atomic

### Performance Considerations

- Use minimal data sizes for unit tests
- Mark slow tests with `@pytest.mark.slow`
- Skip expensive operations when possible (use mocks/stubs)
- Use temporary directories for file operations

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure CODES package is properly installed
2. **Missing datasets**: Some tests require local datasets - download them first
3. **GPU tests failing**: Ensure CUDA is available or skip GPU tests
4. **Network timeouts**: Skip download tests if running without internet

### Debug Mode

Run tests with verbose output:
```bash
pytest test/ -v -s
```

### Test Isolation

Each test should be independent. If tests are interfering with each other:
- Check for global state modifications
- Ensure proper cleanup in fixtures
- Use isolated temporary directories
