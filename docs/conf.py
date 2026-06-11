# docs/source/conf.py

import sys
from pathlib import Path

# sys.path.insert(0, os.path.abspath("../.."))
# using Path.resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
project = "biomassage"
copyright = "2026, umr-lops"
author = "umr-lops"
release = "0.1.0"  # update to your actual version

extensions = [
    "myst_parser",  # for Markdown/MyST support
    "sphinx.ext.autodoc",  # generate API docs from docstrings
    "sphinx.ext.napoleon",  # support NumPy/Google style docstrings
    "sphinx.ext.viewcode",  # add links to source code
    "sphinx.ext.mathjax",  # render math (e.g., $\rightarrow$)
    "sphinx.ext.intersphinx",  # link to external documentation
    # 'nbsphinx',                 # uncomment if you want to execute notebooks
]

# MyST-Parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# For notebook execution (requires jupyter)
# myst_execute_notebooks = 'force'   # or 'cache', 'off'

# Default highlighting for code blocks
highlight_language = "bash"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
