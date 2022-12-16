project = "InVEST urban cooling model calibration"
author = "Martí Bosch"

__version__ = '0.4.1'
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]

autodoc_typehints = "description"
html_theme = "default"
