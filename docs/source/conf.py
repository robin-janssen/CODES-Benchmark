# Configuration file for the Sphinx documentation builder.

import sys
import os

sys.path.insert(0, os.path.abspath('../../'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CODES Benchmark'
copyright = '2024, Robin Janssen, Immanuel Sulzer'
author = 'Robin Janssen, Immanuel Sulzer'
release = '09.09.2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# INFO:
# To generate the rst files automatically, run the following command:
# sphinx-apidoc -o /export/data/isulzer/DON-vs-NODE/docs/test/ /export/data/isulzer/DON-vs-NODE/
#
# To generate the html files, run the following command:
# sphinx-build -M html source build
# sphinx-build -M html /export/data/isulzer/DON-vs-NODE/docs/source /export/data/isulzer/DON-vs-NODE/docs/build
