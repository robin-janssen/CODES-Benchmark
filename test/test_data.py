import glob
from pathlib import Path

import pytest

from codes.utils.data_utils import check_and_load_data, download_data

paths = glob.glob("datasets/*/data.hdf5")
dataset_names = [path.split("/")[1] for path in paths]


@pytest.fixture(params=dataset_names)
def dataset(request: pytest.FixtureRequest):
    return request.param


def test_check_and_load_data(dataset):
    try:
        download_data(dataset)
        _ = check_and_load_data(dataset)
    except Exception as e:
        pytest.fail(f"Failed to load data for {dataset}: {e}")


def test_download_data(dataset, tmp_path: Path):
    path = tmp_path / "data.hdf5"
    try:
        download_data(dataset, path=path)
    except Exception as e:
        pytest.fail(f"Failed to download data for {dataset}: {e}")


def test_download_data_invalid_dataset():
    with pytest.raises(ValueError):
        download_data("invalid_dataset")
