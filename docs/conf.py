# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = "CODES"
copyright = "2024, Robin Janssen, Immanuel Sulzer"
author = "Robin Janssen, Immanuel Sulzer"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_book_theme",
]
autosummary_generate = True
autosummary_generate_overwrite = False
autosummary_output_dir = "_autosummary"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

html_static_path = ["_static"]
html_favicon = "_static/favicon-96x96.png"

# Adjusted HTML theme options
html_theme_options = {
    "icon_links": [
        {
            "name": "CODES GitHub",
            "url": "https://github.com/robin-janssen/CODES-Benchmark",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "CODES Docs",
            "url": "https://codes-docs.web.app",
            "icon": "_static/favicon-96x96.png",  # Make sure this is the correct path to the favicon
            "type": "local",
        },
        # {
        #     "name": "Exemplary Badge",
        #     "url": "https://img.shields.io/badge/Exemplary-Yes-brightgreen",
        #     "icon": "https://img.shields.io/badge/Exemplary-Yes-brightgreen",
        #     "type": "url",
        # },
        # { # Link to the CODES paper once its on arxiv!
        #     "name": "CODES paper",
        #     "url": "https://arxiv.org/abs/2106.04420",
        #     "icon": "fa-file-pdf",
        #     "type": "fontawesome",
        # }
    ],
    "icon_links_label": "Quick Links",
}
