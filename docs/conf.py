# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CODES"
copyright = "2024, Robin Janssen, Immanuel Sulzer"
author = "Robin Janssen, Immanuel Sulzer"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_static_path = ["_static"]
html_favicon = "_static/favicon-96x96.png"
html_theme_options = {
    "external_links": [
        {
            "name": "CODES GitHub",
            "url": "https://github.com/robin-janssen/CODES-Benchmark",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "CODES Docs",
            "url": "../../index.html",
            "icon": "favicon-96x96.png",
            "type": "local",
        },
    ],
    "icon_links": [
        {
            "name": "CODES GitHub",
            "url": "https://github.com/robin-janssen/CODES-Benchmark",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "CODES Docs",
            "url": "../../index.html",
            "icon": "_static/favicon-96x96.png",
            "type": "local",
        },
    ],
    "icon_links_label": "Quick Links",
}
