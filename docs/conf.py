# docs/source/conf.py

import importlib.metadata
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

project = "topsocnww3sp"
copyright = "2026, umr-lops"
author = "umr-lops"


def get_version() -> str:
    """Retrieve the package version dynamically."""
    # 1. Essayer d'importer depuis _version.py (généré par hatch-vcs)
    version_str = None
    try:
        # L'import est local car le fichier peut ne pas exister
        # (par ex. lors d'une compilation sans installation préalable)
        from topsocnww3sp._version import __version__  # noqa: PLC0415

        version_str = __version__
    except ImportError:
        pass
    if version_str is not None:
        return version_str

    # 2. Essayer via importlib.metadata (si le package est installé)
    try:
        return importlib.metadata.version("topsocnww3sp")
    except PackageNotFoundError:
        pass

    # 3. Fallback : récupérer le tag git directement
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--dirty=-dirty"],
            cwd=Path(__file__).parent.parent.parent,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Valeur par défaut si tout échoue
    return "0.0.0+unknown"


release = get_version()

# Extensions Sphinx
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

# Configuration MyST
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

highlight_language = "bash"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
