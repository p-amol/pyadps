# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyadps'
copyright = '2024, p-amol'
author = 'p-amol'

# -- General configuration ---------------------------------------------------

# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'autoapi.extension',
    'myst_nb',
]

autoapi_type = "python"
autoapi_add_toctree_entry = False
autoapi_dirs = ['../../src']
autodoc_mock_imports = ['numpy', 'pandas', 'matplotlib', 'pyadps']
templates_path = ['_templates']
exclude_patterns = []

# Myst-NB configuration
jupyter_execute_notebooks = "off"  # Do not execute cells
execution_excludepatterns = ["**.ipynb"]  # Prevent execution for specific files (if needed)



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_theme_options = {
#     "collapse_navigation": True,  # Optional: Collapse navigation items
#     "navigation_depth": 2,        # Controls ToC depth globally
# }

html_static_path = ['_static']
