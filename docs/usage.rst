.. _usage:

Usage
=====

The ``topsocnww3sp`` package provides a command-line tool ``procl2c`` to co-localize Sentinel-1 OSW data with WW3 spectra.

Running the tool
----------------

Basic usage is as follows:

.. code-block:: bash

    procl2c --ocn-safe /path/to/SAFE --config config.yml --output-dir /path/to/output --mode lasso

Arguments
----------

* ``--ocn-safe`` (Required): Path to the a SAFE directory containing OSW .nc files.
* ``--config`` (Required): Path to the YAML configuration file defining thresholds and directories.
* ``--output-dir`` (Required): Base directory where output files will be saved.
* ``--ww3-file`` (Optional): Specific path or directory of WW3 spectra. If not provided, the tool searches based on SAR time in the config.
* ``--mode`` (Optional): Matching mode. Choices are:
    * ``lasso``: Extracts all spectra within the buffered footprint of the SAR subswath (Default).
    * ``1to1``: One WW3 spectrum per SAR tile (closest match).
    * ``unique``: Multiple SAR tiles can share the same unique WW3 spectrum to save space.
    * ``many``: Records all matches within thresholds.
* ``--verbose`` (Optional): Enable verbose logging.
* ``--overwrite`` (Optional): Overwrite existing output files.

Configuration File
------------------

The YAML config file should contain at least the following keys:

.. code-block:: yaml

    directory_ww3spectra_output: /path/to/ww3/data
    TIME_THRESHOLD_MINUTES: 30
    DISTANCE_THRESHOLD_KM: 14
    product_version: v0.1
