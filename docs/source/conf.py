# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Sistema de Riesgo Crediticio con RBM'
copyright = '2025, Andrés Fernando Gómez Rojas & Carlos Andrés Gómez Vasco. Software Libre bajo Licencia MIT'
author = 'Andrés Fernando Gómez Rojas & Carlos Andrés Gómez Vasco'
release = '1.0.0'

# Información adicional del proyecto
html_context = {
    'author_full': 'Andrés Fernando Gómez Rojas & Carlos Andrés Gómez Vasco',
    'author_institution': 'Universidad Distrital Francisco José de Caldas',
    'author_program': 'Pregrado en Física',
    'director': 'Carlos Andrés Gómez Vasco',
    'version': '1.0.0',
    'license': 'MIT License - Software Libre',
    'year': '2025'
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'es'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Todo extension
todo_include_todos = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}