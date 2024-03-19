# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -*- coding: utf-8 -*-

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os


DOCS_DIR = os.path.split(__file__)[0]


# -- Project information -----------------------------------------------------

project = "PyBalance"
author = "Stephen Privitera, Hooman Sedghamiz, Alex Hartenstein"
copyright = f"2024 - Bayer AG - {author}"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.autosectionlabel",
]

# Add any paths that contain templates here, relative to this directory.
# The master toctree document.
master_doc = "index"
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = [".rst", ".md"]
# -- Options for HTML output -------------------------------------------------
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_title = "PyBalance"
html_theme_options = {"sticky_navigation": True, "display_version": True}
html_show_sourcelink = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# Output file base name for HTML help builder.
htmlhelp_basename = "pybalance"

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
