# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from datetime import datetime
from pathlib import Path

# -- Add root directory to the system path -----------------------------------

sys.path.insert(0, str(Path('..').resolve().absolute()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

date = datetime.now().year
project = 'ArchiTXT'
author = 'Nicolas Hiot, Mirian Halfeld-Ferrari, Jacques Chabin'
copyright = f'{date}, {author}'  # noqa: A001

github_username = 'neplex'
github_repository = 'architxt'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_design',
    'sphinx_copybutton',
    'sphinx_toolbox.more_autodoc',
    'sphinx_toolbox.more_autosummary',
    'sphinx_toolbox.sidebar_links',
    'sphinx_toolbox.github',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc.typehints',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinxcontrib.typer',
    'sphinxcontrib.mermaid',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosectionlabel_prefix_document = True

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'path_to_docs': 'docs',
    'repository_url': f'https://github.com/{github_username}/{github_repository}',
    'use_edit_page_button': True,
    'use_source_button': True,
    'use_issues_button': True,
    'announcement': 'This project is currently in active development. Features may change, and some sections might be incomplete.',
}

# -- Options for autodoc extension -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

autodoc_default_options = {
    'members': True,
    'member-order': 'groupwise',
    'undoc-members': True,
    'show-inheritance': True,
    'ignore-module-all': True,
}

autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True

# -- Options for autosummary extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

autosummary_generate = True

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'nltk': ('https://www.nltk.org/', None),
}

# -- Options for copybutton extension ----------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
