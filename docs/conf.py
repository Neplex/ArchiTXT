# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Add root directory to the system path -----------------------------------

import sys
from pathlib import Path

sys.path.insert(0, str(Path('..').resolve().absolute()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ArchiTXT'
copyright = '2024, Nicolas Hiot, Mirian Halfeld-Ferrari, Jacques Chabin'  # noqa: A001
author = 'Nicolas Hiot, Mirian Halfeld-Ferrari, Jacques Chabin'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autodoc.typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx_rtd_theme',
]

autodoc_typehints = "description"
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
