import os

# use only on IWR Compute Server
os.environ['PIP_CACHE_DIR'] = '/export/scratch2/isulzer/.cache'
os.system('pip install -r requirements.txt')
