#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata

import topsocnww3sp as m


def test_version() -> None:
    assert importlib.metadata.version("topsocnww3sp") == m.__version__
