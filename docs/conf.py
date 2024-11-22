# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys
from textwrap import dedent, indent

import yaml
from sphinx.application import Sphinx
from sphinx.util import logging

LOGGER = logging.getLogger("conf")

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(os.path.abspath('..'))

print("python exec:", sys.executable)
print("sys.path:", sys.path)


project = "ndpyramid"
copyright = "2023, carbonplan"
author = "carbonplan"
release = "v0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
    "sphinx_design",
    "nbsphinx",
]

autosummary_generate = True


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# Sphinx project configuration
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_title = "ndpyarmid"
html_theme_options = {
    "logo": {
        "image_light": "_static/monogram-dark-cropped.png",
        "image_dark": "_static/monogram-light-cropped.png",
    }
}
html_theme = "sphinx_book_theme"
html_title = ""
repository = "carbonplan/ndpyarmid"
repository_url = "https://github.com/carbonplan/ndpyramid"

html_static_path = ["_static"]


def update_gallery(app: Sphinx):
    """Update the gallery page.

    Copied from https://github.com/pydata/xarray/blob/56209bd9a3192e4f1e82c21e5ffcf4c3bacaaae3/doc/conf.py#L399-L430.
    """
    LOGGER.info("Updating gallery page...")

    gallery = yaml.safe_load(pathlib.Path(app.srcdir, "gallery.yml").read_bytes())

    for key in gallery:
        items = [
            f"""
         .. grid-item-card::
            :text-align: center
            :link: {item['path']}

            .. image:: {item['thumbnail']}
                :alt: {item['title']}
            +++
            {item['title']}
            """
            for item in gallery[key]
        ]

        items_md = indent(dedent("\n".join(items)), prefix="    ")
        markdown = f"""
.. grid:: 1 2 2 2
    :gutter: 2

    {items_md}
    """
        pathlib.Path(app.srcdir, f"{key}-gallery.txt").write_text(markdown)
        LOGGER.info(f"{key} gallery page updated.")
    LOGGER.info("Gallery page updated.")


def setup(app: Sphinx):
    app.connect("builder-inited", update_gallery)
