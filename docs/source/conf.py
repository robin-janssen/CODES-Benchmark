from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml

# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = "CODES"
copyright = "2026, Robin Janssen"
author = "Robin Janssen"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_book_theme",
    "myst_nb",
]
autosummary_generate = True
autosummary_generate_overwrite = False
autosummary_output_dir = "_autosummary"

# Render ``Attributes`` sections using the compact parameter style
napoleon_custom_sections = [
    ("Attributes", "params_style"),
]
napoleon_use_ivar = False

# Support RST, Markdown, and notebooks as sources
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Allow Markdown/MyST and notebook content while keeping execution optional
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
]
nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mock optional heavy dependencies for autodoc
autodoc_mock_imports = [
    "optuna",
    "optuna.visualization",
    "optuna.visualization.matplotlib",
    "plotly",
    "plotly.graph_objects",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

html_static_path = ["_static"]
html_favicon = "_static/favicon-96x96.png"
html_logo = "_static/logo.png"

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
            "icon": "_static/favicon-96x96.png", 
            "type": "local",
        },
        {
            "name": "CODES Paper",
            "url": "https://arxiv.org/abs/2410.20886",
            "icon": "fa-solid fa-file-alt",
            "type": "fontawesome",
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


# -- Helpers ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCES = PROJECT_ROOT / "datasets" / "data_sources.yaml"
GENERATED_DATASETS_TABLE = (
    Path(__file__).resolve().parent / "reference" / "_datasets_table.rst"
)


def _clean_download_url(url: str) -> str:
    """Strip the /files/... portion and query params so we link to the dataset landing page."""
    parsed = urlparse(url)
    cleaned_path = parsed.path.split("/files/")[0]
    return f"{parsed.scheme}://{parsed.netloc}{cleaned_path}"


def _write_dataset_table() -> None:
    if not DATA_SOURCES.exists():
        return
    data = yaml.safe_load(DATA_SOURCES.read_text())
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Dataset",
        "     - Source",
    ]
    for name, url in sorted(data.items()):
        cleaned = _clean_download_url(url)
        lines.append(f"   * - ``{name}``")
        lines.append(f"     - `{cleaned} <{cleaned}>`_")
    content = "\n".join(lines) + "\n"
    GENERATED_DATASETS_TABLE.write_text(content)


def _on_builder_inited(app):  # noqa: D401
    _write_dataset_table()


def setup(app):
    app.connect("builder-inited", _on_builder_inited)
