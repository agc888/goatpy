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
    'nbsphinx'
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

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "github_url": "https://github.com/agc888/goatpy",
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",  # swap in a light-colored version if you have one
    },
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_align": "left",
    "show_toc_level": 2,
    "collapse_navigation": False,
    "use_edit_page_button": True,
    "back_to_top_button": True,
}

html_context = {
    "github_user": "agc888",
    "github_repo": "goatpy",
    "github_version": "main",
    "doc_path": "docs/source",   # note: pydata wants doc_path without leading/trailing slashes
}

html_static_path = ['_static']
html_show_sourcelink = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

nbsphinx_execute = 'never'