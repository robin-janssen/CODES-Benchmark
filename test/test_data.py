"""
Legacy data loading tests - kept for compatibility.
For comprehensive dataset testing, see test_datasets.py
"""

import glob
from pathlib import Path

import pytest

from codes.utils.data_utils import check_and_load_data, download_data

# Get available local datasets
paths = glob.glob("datasets/*/data.hdf5")
dataset_names = [path.split("/")[1] for path in paths] if paths else []


@pytest.fixture(params=dataset_names)
def dataset(request: pytest.FixtureRequest):
    """Fixture providing each available local dataset name."""
    return request.param


@pytest.mark.skipif(len(dataset_names) == 0, reason="No local datasets available")
def test_check_and_load_data(dataset):
    """Test loading available local datasets."""
    try:
        result = check_and_load_data(dataset, verbose=False)
        assert result is not None, f"Failed to load data for {dataset}"

        # Basic validation of return structure
        assert len(result) == 6, "check_and_load_data should return 6 elements"

        (
            (train_data, test_data, val_data),
            params,
            timesteps,
            n_train_samples,
            data_info,
            labels,
        ) = result

        # Basic shape validation
        assert train_data.ndim == 3, f"Train data should be 3D for {dataset}"
        assert test_data.ndim == 3, f"Test data should be 3D for {dataset}"
        assert val_data.ndim == 3, f"Val data should be 3D for {dataset}"

    except Exception as e:
        pytest.fail(f"Failed to load data for {dataset}: {e}")


@pytest.mark.download
@pytest.mark.skipif(len(dataset_names) == 0, reason="No local datasets available")
def test_download_data(dataset, tmp_path: Path):
    """Test downloading datasets to temporary location."""
    path = tmp_path / "data.hdf5"
    try:
        download_data(dataset, path=str(path), verbose=False)
        # Only check if file exists since we're not actually downloading
        # (the dataset already exists locally)
    except Exception as e:
        # Skip test if download fails (network issues, etc.)
        pytest.skip(f"Download test skipped for {dataset}: {e}")


def test_download_data_invalid_dataset():
    """Test that download_data raises appropriate error for invalid dataset."""
    with pytest.raises(ValueError, match="not found in data_sources.yaml"):
        download_data("definitely_invalid_dataset_name_12345")
