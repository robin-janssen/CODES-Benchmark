# Dataset Catalog

The repository ships with multiple HDF5 datasets stored under `datasets/<name>/`. Each folder contains `data.hdf5` (train/test/val splits), optional parameter sets, and metadata such as timesteps. The table below is generated automatically from `datasets/data_sources.yaml` so it stays up to date whenever new sources are added.

```{eval-rst}
.. include:: _datasets_table.rst
```

## Download helper

Some datasets are large. Use `python scripts/download_datasets.py --name <dataset>` to fetch a specific file or omit `--name` to download them all. The script reads `datasets/data_sources.yaml`, so keep that file up to date if you mirror the data.
