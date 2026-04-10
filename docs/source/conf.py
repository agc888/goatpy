import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'goatpy'
copyright = '2025, Andrew Causer'
author = 'Andrew Causer'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'autoapi.extension',
]

autoapi_dirs = ['../../goatpy']
autoapi_type = 'python'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]
autoapi_ignore = ['*registration.py', '*tests*']

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'anndata': ('https://anndata.readthedocs.io/en/latest', None),
    'spatialdata': ('https://spatialdata.scverse.org/en/latest', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    "logo_only": True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
}

html_context = {
    "display_github": True,
    "github_user": "agc888",
    "github_repo": "goatpy",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

html_static_path = ['_static']
html_logo = "_static/logo.png"
html_show_sourcelink = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
