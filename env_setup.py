import os

# use only on IWR Compute Server
os.system('export PIP_CACHE_DIR=export/data/scratch.cache')
os.system('pip install -r requirements.txt')