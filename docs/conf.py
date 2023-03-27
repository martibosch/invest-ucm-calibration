"""Docs config."""
project = "InVEST urban cooling model calibration"
author = "Martí Bosch"

__version__ = "0.6.0"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

autodoc_typehints = "description"
html_theme = "default"
