# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sys
import os

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "DeepGEFF"
author = "Roshni Biswas, Anurag Saha Roy, Tejasvi S Tomar"

# The full version, including alpha/beta/rc tags
release = "0.2"


# -- General configuration ---------------------------------------------------

extensions = [
    # Core library for html generation from docstrings
    # "sphinx.ext.autodoc",
    # Create neat summary tables
    # "sphinx.ext.autosummary",
    # Recursive documentation
    "autoapi.extension",
]

autoapi_dirs = ["../../src"]
autoapi_file_patterns = ["*.py"]
autoapi_ignore = ["*logs*", "__pycache__"]
autoapi_member_order = "bysource"
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "inherited-members",
    "special-members",
    "show-inheritance",
    "special-members",
    "imported-members",
    "show-inheritance-diagram",
]

# html_css_files = ["custom.css"]

# Turn on sphinx.ext.autosummary
# autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_1templates"]

# Sort members by type
autodoc_member_order = "groupwise"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["logs", "config"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
