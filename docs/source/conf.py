# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyadps"
copyright = "2024, p-amol"
author = "p-amol"

# -- General configuration ---------------------------------------------------

# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_rtd_theme",  # ReadTheDocs theme
]

# -- MyST configuration ------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

myst_enable_extensions = [
    "colon_fence",  # ::: directive syntax
    "deflist",  # definition lists
    "fieldlist",  # :param: style fields
]

# -- MyST-NB configuration ---------------------------------------------------
nb_execution_mode = "off"  # "auto", "force", "cache", or "off"
nb_execution_excludepatterns = ["**.ipynb"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme_options = {
#     "collapse_navigation": True,  # Optional: Collapse navigation items
#     "navigation_depth": 2,        # Controls ToC depth globally
# }

html_static_path = ["_static"]
