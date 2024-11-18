import os
import sys

# Add the root directory where pyadps is located
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../src/'))
sys.path.insert(0, os.path.abspath('../../src/pyadps'))


project = 'Pyadps'
author = 'amol'
version = '0.1.7'
release = '0.1.7'
language = 'en'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

html_last_updated_fmt = '%b %d, %Y'

autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_mock_imports = ["streamlit","numpy", "pyadps","netCDF4","pandas", "scipy"]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = [
    'https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap',
    'custom.css',
]
html_show_sphinx = False
html_theme_options = {
   "display_version": True,
}

html_context = {
    "copyright": "2024 NIO@amol All rights reserved",
    'display_github': True,
    'last_updated': True,
    'commit': True,
}

