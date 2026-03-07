import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'jax-hod'
copyright = '2025, Enrique Paillas'
author = 'Enrique Paillas'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'myst_parser',
]

source_suffix = ['.rst', '.md']
exclude_patterns = ['_build']

napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_member_order = 'bysource'
autoclass_content = 'both'
autodoc_mock_imports = ['jaxdecomp', 'abacusnbody']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'jax': ('https://jax.readthedocs.io/en/latest', None),
}

# -- HTML output -------------------------------------------------------------
html_theme = 'sphinx_book_theme'
html_title = 'jax-hod'
html_theme_options = {
    'repository_url': 'https://github.com/epaillas/jax-hod',
    'use_repository_button': True,
    'use_issues_button': True,
}
