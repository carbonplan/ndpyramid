# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(os.path.abspath('..'))

print('python exec:', sys.executable)
print('sys.path:', sys.path)


project = 'ndpyramid'
copyright = '2023, carbonplan'
author = 'carbonplan'
release = 'v0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser',
            'sphinx.ext.autodoc',
            'sphinx.ext.autosummary',]

autosummary_generate = True


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# Sphinx project configuration
source_suffix = ['.rst', '.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_title = 'ndpyarmid'
html_theme_options = {
    'logo': {
        'image_light': '_static/monogram-dark-cropped.png',
        'image_dark': '_static/monogram-light-cropped.png',
    }
}
html_theme = 'sphinx_book_theme'
html_title = ''
repository = 'carbonplan/ndpyarmid'
repository_url = 'https://github.com/carbonplan/ndpyramid'

html_static_path = ['_static']
