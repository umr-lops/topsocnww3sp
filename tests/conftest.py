#!/usr/bin/env python3
"""Shared pytest fixtures and import-time setup for the test suite.

Importing ``netCDF4`` here (at collection time) surfaces the one-time
"numpy.ndarray size changed" ABI RuntimeWarning *before* the per-test
``filterwarnings = ["error"]`` filter is applied, so that tests which perform
real NetCDF I/O (e.g. the WW3 FIELD sampling tests) do not fail on a benign
import-time warning emitted by a third-party compiled module.
"""

import netCDF4  # noqa: F401 - imported for its side effect of loading the extension.
