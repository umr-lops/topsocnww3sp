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

    # WW3 FIELD (integrated parameters) configuration. Keys must match the
    # ww3_grids keys so each track point is sampled from its own grid.
    ww3_field_grids:
      arctic:
        pattern: "ARC-*/YYYY-*/FIELD_NC/WW3-ARC-*_*.nc"
      midlatitude:
        pattern: "IRIGLOB-*/YYYY-*/FIELD_NC/WW3-IRIGLOB-*_*.nc"
    # WW3 FIELD variables to sample. The FIELD variable name is preserved as
    # the output name (hs stays hs).
    ww3_field_variables:
      hs:
        attrs:
          long_name: "WW3 significant height of wind and swell waves (total SWH)"
          units: "m"
          standard_name: "sea_surface_wave_significant_height"

Plotting the WW3 Integrated Parameters
--------------------------------------

A standalone script in the ``end2end`` directory renders the WW3 integrated
parameters (e.g. ``hs``) sampled at the track points of an L2C SAFE as a
geographic map. Pass the output SAFE directory: all subswaths (IW1/IW2/IW3) are
drawn on the same map.

.. code-block:: bash

    python end2end/plot_ww3_hs_map_l2c.py \
        --input-l2c-safe /path/to/SAFE_output_dir \
        --output-png ww3_hs_map.png
