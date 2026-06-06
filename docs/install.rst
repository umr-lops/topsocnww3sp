.. _install:

Installation
=============

Installation of ``topsocnww3sp`` can be done via pip from the source repository.

Prerequisites
-------------

- Python 3.10 or higher
- A working Python environment (venv, conda, etc.)

Install from Source
------------------

Clone the repository and install the package:

.. code-block:: bash

    git clone https://github.com/umr-lops/topsocnww3sp.git
    cd topsocnww3sp
    pip install .

Optional Dependencies
---------------------

To enable plotting features, install the optional ``plot`` dependencies:

.. code-block:: bash

    pip install ".[plot]"

Verification
------------

You can verify the installation by checking the version of the installed package:

.. code-block:: bash

    python -c "import topsocnww3sp; print(topsocnww3sp.__version__)"
